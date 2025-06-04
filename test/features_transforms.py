import tempfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from monai.data import create_test_image_3d
from monai.transforms import (
    Compose,
    LoadImage,
)

from toskipornot.features import transforms


def test_transforms():
    root_dir = tempfile.mkdtemp()
    print(root_dir)
    filenames = []

    fn_keys = ("img", "seg")  # filename keys for image and seg files
    filenames = []

    for i in range(5):
        im, seg = create_test_image_3d(256, 256, 256)

        im_filename = f"{root_dir}/im{i}.nii.gz"
        seg_filename = f"{root_dir}/seg{i}.nii.gz"

        filenames.append({"img": im_filename, "seg": seg_filename})

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, im_filename)

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, seg_filename)

    trans = LoadImage()

    img, header = trans(filenames[0])

    print(img.shape, header["filename_or_obj"])
    plt.imshow(img[128])

    trans = Compose([LoadImage(image_only=True), transforms.RandSaltAndPepperNoise(density=0.5)])

    img = trans(filenames[0])

    plt.imshow(img[0, 128])


if __name__ == "__main__":
    test_transforms()
