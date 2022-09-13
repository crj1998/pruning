"""
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

import math

import torch
import torch.nn as nn
import torch.autograd as autograd

class ThresholdBinarizer(autograd.Function):
    """
    Thresholdd binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > \tau`
    where `\tau` is a real value threshold.
    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, sigmoid: bool):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        nb_min = int(0.005 * nb_elems) + 1
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type())
        else:
            mask = (inputs > threshold).type(inputs.type())
        if mask.sum() < nb_min:
            # We limit the pruning so that at least 0.5% (half a percent) of the weights are remaining
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.
    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        idx = inputs.flatten().sort(descending=True).indices
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class MagnitudeBinarizer(object):
    """
    Magnitude Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of |S| (absolute value).
    Implementation is inspired from https://github.com/NervanaSystems/distiller/blob/2291fdcc2ea642a98d4e20629acb5a9e2e04b4e6/distiller/pruning/automated_gradual_pruner.py#L24
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
                This input marix is typically the weight matrix.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        idx = inputs.abs().flatten().sort(descending=True).indices
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            self.mask_scores = nn.Parameter(torch.zeros(self.weight.size()))

            if mask_init == "constant":
                nn.init.constant_(self.mask_scores, val=self.mask_scale)
            elif mask_init == "uniform":
                nn.init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
            elif mask_init == "kaiming":
                nn.init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    def forward(self, inputs: torch.tensor, threshold: float):
        # Get the mask
        if self.pruning_method == "topK":
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in self.pruning_method
            mask = ThresholdBinarizer.apply(self.mask_scores, threshold, sig)
        elif self.pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(self.weight, threshold)
        elif self.pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            eps = 1e-5
            if self.training:
                u = torch.zeros_like(self.mask_scores).uniform_().clamp(eps, 1.0-eps)
                s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.mask_scores) / b)
            else:
                s = torch.sigmoid(self.mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=eps, max=1.0-eps)
        # Compute output (linear layer) with masked weights
        return nn.functional.linear(inputs, mask * self.weight, self.bias)