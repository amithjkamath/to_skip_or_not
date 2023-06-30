"""
This script generates a onnx version of the MONAI Unet to visualize using netron.
"""

from toskipornot.models.NoSkipUnet import NoSkipUNet
from monai.networks.layers import Norm
import torch


def main():
    """
    Generating an ONNX version of the MONAI UNet to visualize using netron.
    """
    model_params = dict(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=0,
        norm=Norm.BATCH,
    )

    x = torch.randn(1, 1, 96, 96, requires_grad=True)
    model = NoSkipUNet(**model_params)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "noskipunet_monai.onnx",  # where to save the model (can be a file or file-like object)
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
