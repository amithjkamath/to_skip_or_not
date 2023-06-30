"""
This script generates a onnx version of the MONAI Unet to visualize using netron.
"""

from monai.networks.nets import VNet
from monai.networks.layers import Norm
import torch


def main():
    """
    Generating an ONNX version of the MONAI UNet to visualize using netron.
    """
    model_params = dict(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        act="ReLU",
        dropout_prob=0.0,
    )

    x = torch.randn(1, 1, 96, 96, requires_grad=True)
    model = VNet(**model_params)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "vnet_monai.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
