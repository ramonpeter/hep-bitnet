import torch
from torch import Tensor, nn

torch.manual_seed(6)


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
        self.eps = 1e-8
        self.norm = nn.LayerNorm(in_features)

        # Quantiziation and dequantization
        self.Q_b = 2 ** (b - 1)  # use this to define quantized bit
        self.beta = torch.tensor(
            0.0, device=self.weight.device, dtype=self.weight.dtype
        )
        self.gamma = torch.tensor(
            0.0, device=self.weight.device, dtype=self.weight.dtype
        )

    def ste(self, x):
        """
        Applies the sign function for binarization and uses Straight-Through Estimator (STE) during backward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Binarized tensor.
        """
        binarized_x = torch.sign(x)
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights(self):
        """
        Binarizes the weights of the layer using STE.

        Returns:
            Tensor: Binarized weights tensor.
        """
        alpha = self.weight.mean()
        self.beta = torch.maximum(self.weight.abs().mean(), torch.tensor(self.eps))
        binarized_weights = self.ste(self.weight - alpha)

        return binarized_weights

    def quantize_activations(self, x):
        """
        Quantizes the activations of the layer.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        self.gamma = x.abs().max()
        quantized_x = torch.clamp(
            x * self.Q_b / (self.gamma + self.eps),
            -self.Q_b + self.eps,
            self.Q_b - 1.0 - self.eps,
        )
        return quantized_x

    def dequantize_activations(self, x):
        """
        Dequantizes the activations of the layer.

        Args:
            x (Tensor): Quantized input tensor.

        Returns:
            Tensor: Dequantized activations tensor.
        """
        return x * self.gamma * self.beta / self.Q_b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Normalize input
        x = self.norm(x)

        # Binarize weights and quantize activations
        binarized_weights = self.binarize_weights()

        # Quantize input
        x_quant = self.quantize_activations(x)

        # Perform linear transformation
        output = torch.nn.functional.linear(x_quant, binarized_weights, self.bias)

        # Dequantize activations
        output = self.dequantize_activations(output)

        # Return output
        return output


class BitLinear158b(BitLinear):
    """
    BitLinear158b layer allowing for tertiar weights (-1,0,1). Rest is keeped
    as in BitLinear

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_groups (int, optional): Number of groups to divide the weights and activations into. Default is 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias, b)

    def _absmean_quantization(self, weight, gamma):
        quantized_weight = torch.clamp(
            torch.round(weight / (gamma + self.eps)), min=-1, max=1
        )
        quantized_weight = (quantized_weight - weight).detach() + weight
        return quantized_weight

    def binarize_weights(self):
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        self.beta = torch.maximum(self.weight.abs().mean(), torch.tensor(self.eps))
        binarized_weight = self._absmean_quantization(self.weight, self.beta)

        return binarized_weight


if __name__ == "__main__":
    # Example usage
    bitlinear = BitLinear(3, 4)
    bitlinear2 = BitLinear(4, 2)
    input_tensor = torch.randn(2, 3, requires_grad=True)  # Example input tensor
    output = bitlinear2(bitlinear(input_tensor)).sum()
    # print(output)  # Example output tensor
    output.backward()
    # Access the gradients using x.grad
    dx = input_tensor.grad
    print("x.grad :", dx)
