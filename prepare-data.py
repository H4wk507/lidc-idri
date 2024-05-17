import os

import matplotlib.pyplot as plt
import pylidc as pl


# TODO: test on whole lidc dataset 100GB
def save_slices_and_masks() -> None:
    """Save imgs and masks with subtlety >= 5 as 512x512 images in ./data."""

    os.makedirs("./data/imgs", exist_ok=True)
    os.makedirs("./data/masks", exist_ok=True)

    plt.set_cmap("gray")
    padding = [(512, 512), (512, 512), (0, 0)]

    annos = pl.query(pl.Annotation).filter(pl.Annotation.subtlety >= 5)
    nannos = annos.count()

    for anno_idx, anno in enumerate(annos[1587:]):
        try:
            img = anno.scan.to_volume()
            mask = anno.boolean_mask(pad=padding)
            for slice_idx, slice in enumerate(anno.contour_slice_indices):
                if slice_idx >= mask.shape[2]:
                    break
                plt.imsave(f"./data/imgs/img_{anno_idx}_{slice_idx}.png", img[:, :, slice])
                plt.imsave(f"./data/masks/mask_{anno_idx}_{slice_idx}.png", mask[:, :, slice_idx])
        except Exception as e:
            print(f"Error: {e}")
        print(f"Iteration {anno_idx+1}/{nannos}")
    print("Done.")

if __name__ == "__main__":
    save_slices_and_masks()
