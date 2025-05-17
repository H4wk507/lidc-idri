## Download dataset

To download full 40GB dataset follow this:

1. Download 'archive.zip' from https://www.kaggle.com/datasets/washingtongold/lidcidri30
2. Unzip it into lidc directory (this could take a while)

```bash
unzip archive.zip -d lidc
```

Run prepare data script to convert the dataset into a format that can be used by the model.

Our prepare-data.py script uses pylidc which uses some old dependencies. With python 3.9 everything works fine.
But for model training/inference you should use latest python version.

```bash
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install -r prepare-data-requirements.txt
python3.9 prepare-data.py
```

Or you can use already prepared zip of 3000 images to save time `data.zip`.

## References

- [Road Extraction by Deep Residual U-Net](https://arxiv.org/pdf/1711.10684)
- [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592)
- [ResUNet++: An Advanced Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1911.07067)
