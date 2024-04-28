## Download dataset

1. Download 'archive.zip' from https://www.kaggle.com/datasets/washingtongold/lidcidri30
2. Unzip it into lidc directory (this could take a while)

```bash
unzip archive.zip -d lidc
```

## Install dependencies

Some old dependenceis won't run with newer versions of numpy and python.
In python 3.9 everything works fine.

```bash
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install -r requirements.txt
```

## Other datasets

There is a dataset https://www.kaggle.com/datasets/zhangweiled/lidcidri/data that has already extracted nodules into png images and their masks,
so maybe we could use that one for classifiction and segmentation, but research is needed.
