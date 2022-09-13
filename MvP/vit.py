import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (BaseModelOutput, BaseModelOutputWithPooling)
from transformers.models.vit.modeling_vit import ViTModel, ViTForImageClassification, ImageClassifierOutput, ViTEmbeddings, ViTEncoder, ViTLayer, ViTIntermediate, ViTOutput, ViTAttention, ViTSelfAttention, ViTSelfOutput


from mvp import MaskedLinear

class MoveViTForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = MoveViTModel(config, add_pooling_layer=False)
        self.post_init()

    def forward(
        self,
        pixel_values = None,
        head_mask = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        interpolate_pos_encoding = None,
        return_dict = True,
        threshold=1.0,
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
            threshold=threshold,
        )

        sequence_output = outputs[0]
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


class MoveViTModel(ViTModel):
    def __init__(self, config, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = MoveViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        threshold=1.0,
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

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            threshold=threshold
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


class MoveViTEncoder(ViTEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([MoveViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        threshold=1.0
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
                        return module(*inputs, output_attentions, threshold=threshold)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask,
                    output_attentions, threshold=threshold
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


class MoveViTLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.attention = MoveViTAttention(config) # TODO
        self.output = MoveViTOutput(config)
        self.intermediate = MoveViTIntermediate(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        threshold=1.0
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            threshold=threshold
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output, threshold=threshold)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states, threshold=threshold)

        outputs = (layer_output,) + outputs

        return outputs


class MoveViTAttention(ViTAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MoveViTSelfAttention(config)
        self.output = MoveViTSelfOutput(config)

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        threshold=1.0
    ):
        self_outputs = self.attention(
            hidden_states,
            head_mask,
            output_attentions,
            threshold=threshold
        )

        attention_output = self.output(self_outputs[0], hidden_states, threshold=threshold)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MoveViTSelfAttention(ViTSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.query = MaskedLinear(
            config.hidden_size,
            self.all_head_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )
        self.key = MaskedLinear(
            config.hidden_size,
            self.all_head_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )
        self.value = MaskedLinear(
            config.hidden_size,
            self.all_head_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )


    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        threshold=1.0
    ):
        mixed_query_layer = self.query(hidden_states, threshold=threshold)

        key_layer = self.transpose_for_scores(self.key(hidden_states, threshold=threshold))
        value_layer = self.transpose_for_scores(self.value(hidden_states, threshold=threshold))
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

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class MoveViTSelfOutput(ViTSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = MaskedLinear(
            config.hidden_size,
            config.hidden_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )

    def forward(self, hidden_states, input_tensor, threshold=1.0):
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MoveViTIntermediate(ViTIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.dense = MaskedLinear(
            config.hidden_size,
            config.intermediate_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )

    def forward(self, hidden_states: torch.Tensor, threshold=1.0) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class MoveViTOutput(ViTOutput):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dense = MaskedLinear(
            config.intermediate_size,
            config.hidden_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )

    def forward(self, hidden_states, input_tensor, threshold=1.0):
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states


if __name__ == '__main__':
    from transformers import ViTConfig
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    
    # del config.label2id
    # del config.id2label
    config.pruning_method = "topK"
    config.mask_init = "constant"
    config.mask_scale = 0.0

    x = torch.rand(1, 3, 224, 224)
    model = MoveViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = MoveViTForImageClassification(config)

    with torch.no_grad():
        # y = model(x, output_attentions=True, output_hidden_states=True)
        y = model(x)
        print(y)
