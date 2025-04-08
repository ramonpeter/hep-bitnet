import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vblinear import VBLinear


class BitLinear(nn.Linear):
    """
    BitLinear is a custom linear layer that performs quantization of weights and activations

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-8
        self.device = self.weight.device
        self.dtype = self.weight.dtype

        # Quantiziation and dequantization
        self.Q_b = 2 ** (b - 1) - 1.0
        self.beta = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.gamma = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def quantize_weights(self, w: Tensor) -> Tensor:
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        alpha = w.mean()
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = torch.sign(w - alpha)

        return quantized_weight * self.beta

    def quantize_activations(self, x: Tensor) -> Tensor:
        """
        Quantizes the activations of the layer.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        self.gamma = self.Q_b / x.abs().max(dim=-1, keepdim=True).values.clamp_(
            min=self.eps
        )
        quantized_x = (x * self.gamma).round().clamp_(-(self.Q_b + 1), self.Q_b)

        return quantized_x / self.gamma

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # weight tensor with shape (in_features, out_features)
        w = self.weight

        # Quantize weights
        w_quant = w + (self.quantize_weights(w) - w).detach()

        # Quantize input
        x_quant = x + (self.quantize_activations(x) - x).detach()

        # Perform linear transformation
        output = F.linear(x_quant, w_quant, self.bias)

        # Return dequantized output
        return output


class BitLinear158b(BitLinear):
    """
    BitLinear158b layer allowing for tertiar weights (-1,0,1). Rest is keeped
    as in BitLinear

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias, b)

    def quantize_weights(self, w: Tensor):
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = (w / self.beta).round().clamp_(-1, 1)

        return quantized_weight * self.beta


class VBBitLinear(VBLinear):
    """
    Bayesian version of the BitLinear layer

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-8
        self.device = self.weight.device
        self.dtype = self.weight.dtype

        # Quantiziation and dequantization
        self.Q_b = 2 ** (b - 1) - 1.0
        self.beta = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.gamma = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def quantize_weights(self, w: Tensor) -> Tensor:
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        alpha = w.mean()
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = torch.sign(w - alpha)

        return quantized_weight * self.beta

    def quantize_activations(self, x: Tensor) -> Tensor:
        """
        Quantizes the activations of the layer.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        self.gamma = self.Q_b / x.abs().max(dim=-1, keepdim=True).values.clamp_(
            min=self.eps
        )
        quantized_x = (x * self.gamma).round().clamp_(-(self.Q_b + 1), self.Q_b)

        return quantized_x / self.gamma

    def forward(self, inpt):
        if self.resample:
            self.random = torch.randn_like(self.logsig2_w)
        s2_w = self.logsig2_w.exp()
        weight = self.mu_w + s2_w.sqrt() * self.random
        return nn.functional.linear(inpt, weight, self.bias)  # + 1e-8

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Quantize input
        x_quant = x + (self.quantize_activations(x) - x).detach()

        if self.map:
            # Quantize mean weights only
            mu_w = self.mu_w
            mu_quant = mu_w + (self.quantize_weights(mu_w) - mu_w).detach()
            return F.linear(x_quant, mu_quant, self.bias)

        logsig2_w = self.logsig2_w.clamp(-11, 11)
        s2_w = logsig2_w.exp()

        if self.random is None:
            self.random = torch.randn_like(self.logsig2_w)

        # Quantize full weight
        w = self.mu_w + s2_w.sqrt() * self.random
        w_quant = w + (self.quantize_weights(w) - w).detach()

        # Perform linear transformation
        output = F.linear(x_quant, w_quant, self.bias) + 1e-8

        # Return dequantized output
        return output


class VBBitLinear158b(VBBitLinear):
    """
    Bayesian version of BitLinear158b layer

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias, b)

    def quantize_weights(self, w: Tensor):
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = (w / self.beta).round().clamp_(-1, 1)

        return quantized_weight * self.beta
