## Download dataset

1. Download 'archive.zip' from https://www.kaggle.com/datasets/washingtongold/lidcidri30
2. Unzip it into lidc directory (this could take a while)

```bash
unzip archive.zip -d lidc
```

## Install dependencies

Some old dependencies won't run with newer versions of numpy and python.
In python 3.9 everything works fine.

```bash
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install -r requirements.txt
```

## References

- [Road Extraction by Deep Residual U-Net](https://arxiv.org/pdf/1711.10684)
- [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592)
- [ResUNet++: An Advanced Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1911.07067)
