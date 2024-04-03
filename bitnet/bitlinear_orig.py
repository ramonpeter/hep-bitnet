import torch
from torch import Tensor, nn


class BitLinear(nn.Linear):
    """
    BitLinear is a custom linear layer that performs binarization of weights and quantization of activations

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-5
        self.norm = nn.LayerNorm(in_features)

        # Quantiziation and dequantization
        self.Q_b = torch.tensor(
            2 ** (b - 1) - 1.0, device=self.weight.device, dtype=self.weight.dtype
        )
        self.beta = torch.tensor(
            0.0, device=self.weight.device, dtype=self.weight.dtype
        )
        self.gamma = torch.tensor(
            0.0, device=self.weight.device, dtype=self.weight.dtype
        )

    def quantize_weights(self, w: torch.Tensor) -> Tensor:
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        alpha = w.mean()
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = torch.sign(w - alpha)

        return quantized_weight * self.beta

    def quantize_activations(self, x: torch.Tensor) -> Tensor:
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
        quantized_x = (x / self.gamma).round().clamp_(-128, 127)

        return quantized_x * self.gamma

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
        output = torch.nn.functional.linear(x_quant, w_quant, self.bias)

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
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias, b)

    def quantize_weights(self, w: torch.Tensor):
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = (w / self.beta).round().clamp_(-1, 1)

        return quantized_weight * self.beta


if __name__ == "__main__":
    # Example usage
    bitlinear = BitLinear158b(3, 5)
    bitlinear2 = BitLinear158b(5, 2)
    input_tensor = torch.randn(3, 3, requires_grad=True)  # Example input tensor
    output = bitlinear2(bitlinear(input_tensor)).sum()
    # output2 = bitlinear2(bitlinear(input_tensor, train=False), train=False).sum()
    print(output)  # Example output tensor
    # Test evaluation step
    # Access the gradients using x.grad
    output.backward()
    dx = input_tensor.grad
    print("x.grad :", dx)
