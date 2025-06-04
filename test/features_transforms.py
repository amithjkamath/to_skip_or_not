import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from monai.data import create_test_image_3d
from monai.transforms import (
    Compose,
    LoadImage,
)

from toskipornot.features import transforms


def features_transforms():
    root_dir = "/Users/amithkamath/repo/to_skip_or_not/data"
    os.makedirs(root_dir, exist_ok=True)
    print(root_dir)

    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(root_dir, f"im{i}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(root_dir, f"seg{i}.nii.gz"))

    images = sorted(glob.glob(os.path.join(root_dir, "im*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(root_dir, "seg*.nii.gz")))

    imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            transforms.RandSaltAndPepperNoise(density=0.5),
        ]
    )

    img, header = imtrans(images[0])

    print(img.shape, header["filename_or_obj"])

    plt.imshow(img[0, 128])


if __name__ == "__main__":
    features_transforms()
