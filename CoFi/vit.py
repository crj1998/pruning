import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (BaseModelOutput, BaseModelOutputWithPooling)
from transformers.models.vit.modeling_vit import ViTModel, ViTForImageClassification, ImageClassifierOutput, ViTEmbeddings, ViTEncoder, ViTLayer, ViTIntermediate, ViTOutput, ViTAttention, ViTSelfAttention, ViTSelfOutput

class CoFiViTForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = CoFiViTModel(config, add_pooling_layer=False)
        # self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size) if layer_distill else None
        self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        pixel_values = None,
        head_mask = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        interpolate_pos_encoding = None,
        return_dict = True,
        head_layer_z=None,
        head_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            head_layer_z=head_layer_z,
            head_z=head_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )

        sequence_output = outputs[0]
        if hidden_z is not None:
            sequence_output = sequence_output.mul(hidden_z)

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CoFiViTModel(ViTModel):
    def __init__(self, config, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = CoFiViTEmbeddings(config)
        self.encoder = CoFiViTEncoder(config)
        self.layernorm = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.embeddings = CoFiViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = CoFiViTEncoder(config)

        self.layernorm = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=True,
        head_layer_z=None,
        head_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding, hidden_z=hidden_z)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, hidden_z=hidden_z)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


class CoFiLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output


class CoFiViTEmbeddings(ViTEmbeddings):
    """ Inherit from BertEmbeddings to allow CoFiLayerNorm """

    def __init__(self, config, use_mask_token: bool = False):
        super().__init__(config, use_mask_token)

    def forward(self, 
        pixel_values, 
        bool_masked_pos=None,
        interpolate_pos_encoding=False,
        hidden_z=None
    ):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)

        embeddings = self.dropout(embeddings)

        return embeddings


class CoFiViTEncoder(ViTEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CoFiViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            output_attentions,
                            intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                            head_z=head_z[i] if head_z is not None else None,
                            mlp_z=mlp_z[i] if mlp_z is not None else None,
                            head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                            hidden_z=hidden_z,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                    head_z=head_z[i] if head_z is not None else None,
                    mlp_z=mlp_z[i] if mlp_z is not None else None,
                    head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                    hidden_z=hidden_z,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CoFiViTLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.attention = CoFiViTAttention(config)
        self.output = CoFiViTOutput(config)
        self.intermediate = ViTIntermediate(config)
        self.layernorm_before = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states, hidden_z=hidden_z),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states, hidden_z=hidden_z)

        layer_output = self.intermediate(layer_output)
        if intermediate_z is not None:
            layer_output = layer_output.mul(intermediate_z)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class CoFiViTAttention(ViTAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CoFiViTSelfAttention(config)
        self.output = CoFiViTSelfOutput(config)

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None
    ):
        self_outputs = self.attention(
            hidden_states,
            head_mask,
            output_attentions,
            head_z=head_z
        )

        attention_output = self.output(self_outputs[0], hidden_states, head_layer_z=head_layer_z, hidden_z=hidden_z)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CoFiViTSelfAttention(ViTSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self,
                hidden_states,
                head_mask=None,
                output_attentions=False,
                head_z=None):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if head_z is not None:
            context_layer = context_layer.mul(head_z)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class CoFiViTSelfOutput(ViTSelfOutput):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, input_tensor, head_layer_z=None, hidden_z=None, inference=False):
        if hidden_states is None:
            return input_tensor
        hidden_states = self.dense(hidden_states)
        if head_layer_z is not None:
            hidden_states = hidden_states.mul(head_layer_z)
        if not inference and hidden_states.sum().eq(0).item():
            hidden_states = hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states


class CoFiViTOutput(ViTOutput):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, hidden_states, input_tensor, mlp_z=None, hidden_z=None, inference=False):
        hidden_states = self.dense(hidden_states)
        if mlp_z is not None:
            hidden_states = hidden_states.mul(mlp_z)
        if not inference and hidden_states.sum().eq(0).item():
            return hidden_states + input_tensor
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


if __name__ == '__main__':
    from l0module import L0Module

    x = torch.rand(1, 3, 224, 224)

    # model = CoFiViTForImageClassification.from_pretrained('test-cifar-10/checkpoint-1056')
    model = CoFiViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    l0module = L0Module(model.config)
    zs = l0module(True)

    with torch.no_grad():
        y = model(x, **zs, output_attentions=True, output_hidden_states=True)
        # y = model(x)
        print(y)

    # from transformers import ViTModel, ViTConfig

    # x = torch.rand(1, 3, 224, 224)

    # # Initializing a ViT vit-base-patch16-224 style configuration
    # cfg = {
    #     'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12, 
    #     'intermediate_size': 768*4, 'hidden_act': 'gelu', 
    #     'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0, 'initializer_range': 0.02, 
    #     'layer_norm_eps': 1e-12, 'image_size': 224, 'patch_size': 16, 
    #     'num_channels': 3, 'qkv_bias': True, 'encoder_stride': 16
    # }
    # cfg.update({
    #     'hidden_size': 192, 
    #     'num_hidden_layers': 12,
    #     'num_attention_heads': 3,
    #     'intermediate_size': 4*192
    # })
    # config = ViTConfig(**cfg)

    # # Initializing a model from the vit-base-patch16-224 style configuration
    # # model = ViTModel(config)
    # # model = CoFiViTModel(config)
    # model = CoFiViTForImageClassification(config)
    # l0module = L0Module(config)
    # zs = l0module(True)
    # # print(zs)
    # # Accessing the model configuration
    # print(model.config)
    # print(sum(p.numel() for p in model.parameters()))

    # with torch.no_grad():
    #     y = model(x, **zs)

    # model = CoFiViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # print(vars(model.config).keys())
    # l0module = L0Module(model.config)
    # zs = l0module(True)

    # with torch.no_grad():
    #     y = model(x, **zs)
    #     y = model(x)