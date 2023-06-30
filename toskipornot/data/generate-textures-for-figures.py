"""
generate-random-textures uses textures from https://github.com/boschresearch/GridSaliency-ToyDatasetGen to 
generate random crops with binary mask labels from generate-random-mask-shapes.py
"""


import os
import numpy as np
from PIL import Image, ImageChops
from random import randrange


if __name__ == "__main__":
    root_path = "/Users/amithkamath/repo/toskipornot"
    mask_path = os.path.join(root_path, "data", "raw")
    texture_path = os.path.join(root_path, "data", "textures")
    out_folder_name = "paper-figures"

    # for shape in ["easiest", "easier", "normal", "harder", "hardest"]:
    for shape in ["normal"]:
        mask_path = os.path.join(mask_path, "masks_" + shape)

        texture_bg_img = Image.open(os.path.join(texture_path, "'fb2dd8bc.png"))
        texture_fg_img = Image.open(os.path.join(texture_path, "'2fbd466c.png"))

        img_size = 256

        wpercent = 2 * img_size / float(texture_bg_img.size[0])
        hsize = int((float(texture_bg_img.size[1]) * float(wpercent)))
        texture_bg_img = texture_bg_img.resize(
            (2 * img_size, hsize), Image.Resampling.LANCZOS
        )

        wpercent = 2 * img_size / float(texture_fg_img.size[0])
        hsize = int((float(texture_fg_img.size[1]) * float(wpercent)))
        texture_fg_img = texture_fg_img.resize(
            (2 * img_size, hsize), Image.Resampling.LANCZOS
        )

        for prop in [10, 20, 30, 40, 50, 60, 70, 80, 82, 85, 88, 90, 92, 95, 98]:
            dataset_path = os.path.join(
                root_path,
                "data",
                out_folder_name,
                "alphablend_0p" + str(prop) + "_" + shape,
            )

            os.makedirs(os.path.join(dataset_path), exist_ok=True)

            for i in range(1):
                x, y = texture_bg_img.size
                x_bg = randrange(0, x - img_size)
                y_bg = randrange(0, y - img_size)
                bg_img = texture_bg_img.crop(
                    (x_bg, y_bg, x_bg + img_size, y_bg + img_size)
                )
                bg_arr = np.asarray(bg_img).copy().astype(np.float64)

                x, y = texture_fg_img.size
                x_fg = randrange(0, x - img_size)
                y_fg = randrange(0, y - img_size)

                fg_img = texture_fg_img.crop(
                    (x_fg, y_fg, x_fg + img_size, y_fg + img_size)
                )
                fg_arr = np.asarray(fg_img).copy().astype(np.float64)

                mask_img = Image.open(
                    os.path.join(mask_path, "mask_" + str(i).zfill(3) + ".png")
                )
                mask_arr = np.asarray(mask_img).copy().astype(np.float64)

                blended_fg = Image.blend(fg_img, bg_img, prop / 100)
                train_img = Image.composite(blended_fg, bg_img, mask_img)

                train_img.save(
                    os.path.join(
                        dataset_path,
                        "img_" + str(i).zfill(4) + ".png",
                    )
                )
                mask_img.save(
                    os.path.join(
                        dataset_path,
                        "mask_" + str(i).zfill(4) + ".png",
                    )
                )
                fg_only = ImageChops.multiply(train_img, mask_img)
                fg_only.save(
                    os.path.join(
                        dataset_path,
                        "fg_only_" + str(i).zfill(4) + ".png",
                    )
                )
                bg_only = ImageChops.multiply(train_img, ImageChops.invert(mask_img))
                bg_only.save(
                    os.path.join(
                        dataset_path,
                        "bg_only_" + str(i).zfill(4) + ".png",
                    )
                )
