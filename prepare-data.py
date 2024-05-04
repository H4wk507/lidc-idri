import os

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl


# Unused
def save_nodules() -> None:
    """Save nodules with subtlety >= 3 as 50x50 images in ./nodules."""
    annos = pl.query(pl.Annotation).filter(pl.Annotation.subtlety >= 3)

    for i in range(annos.count()):
        anno = annos[i]
        os.makedirs("./nodules", exist_ok=True)
        vol = anno.scan.to_volume()
        masks = anno.boolean_mask()

        # Make sure images are 50x50

        top_pad = (50 - masks.shape[0]) // 2
        bottom_pad = 50 - masks.shape[0] - top_pad

        left_pad = (50 - masks.shape[1]) // 2
        right_pad = 50 - masks.shape[1] - left_pad

        padding = [
            (top_pad, bottom_pad),
            (left_pad, right_pad),
            (0, 0),
        ]

        bbox = anno.bbox(pad=padding)
        relevant_vol = vol[bbox]

        slices_indices = anno.contour_slice_indices
        print(slices_indices)
        plt.set_cmap("gray")
        for region in range(masks.shape[2]):
            plt.axis("off")
            plt.imsave(f"./nodules/{i}_{region}.png", relevant_vol[:, :, region])


def save_slices_and_masks(nannos: int) -> None:
    """Save slices and masks to d.npy file."""
    imgs = []
    masks = []

    plt.set_cmap("gray")
    padding = [(512, 512), (512, 512), (0, 0)]

    annos = pl.query(pl.Annotation).filter(pl.Annotation.subtlety == 5)[:nannos]
    for anno_idx, anno in enumerate(annos):
        img = anno.scan.to_volume()
        mask = anno.boolean_mask(pad=padding)
        for slice_idx, slice in enumerate(anno.contour_slice_indices):
            if slice_idx >= mask.shape[2]:
                break
            masks.append(mask[:, :, slice_idx])
            imgs.append(img[:, :, slice])
        print(f"Iteration {anno_idx+1}/{nannos}")

    data = {"imgs": np.array(imgs), "masks": np.array(masks)}
    np.save("d.npy", data)


def main() -> None:
    save_slices_and_masks(250)


if __name__ == "__main__":
    main()
