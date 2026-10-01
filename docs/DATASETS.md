# Datasets

CIFA is evaluated on four standard DomainBed benchmarks under
leave-one-domain-out (LODO):

| Dataset | Domains | Classes | Backbone |
|---|---|---|---|
| PACS | art_painting, cartoon, photo, sketch | 7 | ResNet-50 |
| VLCS | CALTECH, LABELME, PASCAL, SUN | 5 | ResNet-50 |
| OfficeHome | Art, Clipart, Product, Real World | 65 | ResNet-50 |
| TerraIncognita | location_38, location_43, location_46, location_100 | 10 | ResNet-50 |

The code **does not download anything**. Obtain the datasets from their
official releases (or the DomainBed download helper,
`python -m domainbed.scripts.download --data_dir <dir>`, using
<https://github.com/facebookresearch/DomainBed>) and arrange them in the
ImageFolder layouts below. Then pass the root with `DATA_DIR` (or set `data_dir`
in the corresponding `configs/*.yaml`).

## PACS

```
<PACS root>/
├── art_painting/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
├── cartoon/...
├── photo/...
└── sketch/...
```

The seven class folders must be identical across all four domains. Example:

```bash
DATA_DIR=/data/PACS bash scripts/run_sweep.sh pacs cifa
```

## VLCS

This release nests every domain under three fixed sub-splits; the loader merges
`train`, `test` and `crossval` (an optional `full` subset, if present, is
ignored). The five classes are the `0..4` folders inside each split:

```
<VLCS root>/
├── CALTECH/{train,test,crossval}/{0,1,2,3,4}/*.jpg
├── LABELME/...
├── PASCAL/...
└── SUN/...
```

## OfficeHome

```
<OfficeHome root>/
├── Art/<65 class folders>/*.jpg
├── Clipart/...
├── Product/...
└── Real World/...        # on disk this may be spelled Realworld or Real_World
```

The loader resolves the fourth domain case- and separator-insensitively, so
`Real World`, `Realworld`, `Real_World` and `RealWorld` all work; keep the
canonical name `Real World` in configs and commands.

## TerraIncognita

The raw ECCV-2018 release is a flat image folder with COCO-style annotations;
convert it to an ImageFolder tree (symlinks, original files untouched):

```bash
python scripts/prepare_terra.py \
    --ann  <raw>/eccv_18_annotation_files \
    --img  <raw>/eccv_18_all_images_sm \
    --out  <raw>/terra_imagefolder
```

This produces, keeping locations 38/43/46/100 and the ten wildlife classes
(`bird, bobcat, cat, coyote, dog, empty, opossum, rabbit, raccoon, squirrel`):

```
terra_imagefolder/
├── location_38/<class>/*.jpg
├── location_43/...
├── location_46/...
└── location_100/...
```

## Verifying a dataset

A quick sanity check before a full sweep:

```bash
cd cifa
python -c "from datasets import get_dataset; \
d=get_dataset('pacs','/data/PACS','sketch',0.2,0); \
print('classes:', d['classes']); \
print('train imgs:', len(d['train_dataset']))"
```

The printed class list/order must match across domains; a mismatch (e.g. a
missing class folder in one domain) raises `target class ordering mismatch`.
