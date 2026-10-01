#!/usr/bin/env python3
"""Convert the raw ECCV-2018 TerraIncognita release into a DomainBed-aligned
ImageFolder tree (4 location domains x 10 classes), using symlinks.

Raw layout (flat images + COCO annotations split by time):
  eccv_18_all_images_sm/<uuid>.jpg
  eccv_18_annotation_files/{train,cis_val,cis_test,trans_val,trans_test}_annotations.json

Output:
  terra_imagefolder/location_<38|43|46|100>/<class>/<uuid>.jpg  (symlinks)
Original files are never moved or modified.
"""
import argparse
import json
import os

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--ann",
               default="/mnt/sdd1/datasets_qsong/TerraIncognita/eccv_18_annotation_files",
               help="directory containing the *_annotations.json files")
p.add_argument("--img",
               default="/mnt/sdd1/datasets_qsong/TerraIncognita/eccv_18_all_images_sm",
               help="directory containing the raw <uuid>.jpg images")
p.add_argument("--out",
               default="/mnt/sdd1/datasets_qsong/TerraIncognita/terra_imagefolder",
               help="output ImageFolder root (location_*/<class>/)")
args = p.parse_args()
ANN, IMG, OUT = args.ann, args.img, args.out
FILES = ["train_annotations.json", "cis_val_annotations.json",
         "cis_test_annotations.json", "trans_val_annotations.json",
         "trans_test_annotations.json"]
KEEP = [38, 43, 46, 100]
TEN = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty',
       'opossum', 'rabbit', 'raccoon', 'squirrel']

images, anns, cats = {}, {}, {}
for fn in FILES:
    d = json.load(open(os.path.join(ANN, fn)))
    for c in d["categories"]:
        cats[c["id"]] = c["name"]
    for im in d["images"]:
        if im["id"] not in images:      # union, first occurrence wins
            images[im["id"]] = im
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

os.makedirs(OUT, exist_ok=True)
counts = {loc: {c: 0 for c in TEN} for loc in KEEP}
dropped = 0
missing = 0
for im in images.values():
    loc = im.get("location")
    if loc not in KEEP:
        continue
    cls = None
    for a in anns.get(im["id"], []):     # annotation order, first keep class
        name = cats[a["category_id"]]
        if name in TEN:
            cls = name
            break
    if cls is None:
        dropped += 1
        continue
    src = os.path.join(IMG, im["file_name"])
    if not os.path.exists(src):
        missing += 1
        continue
    ddir = os.path.join(OUT, f"location_{loc}", cls)
    os.makedirs(ddir, exist_ok=True)
    link = os.path.join(ddir, im["file_name"])
    if not os.path.lexists(link):
        os.symlink(src, link)
    counts[loc][cls] += 1

total = 0
for loc in KEEP:
    rt = sum(counts[loc].values())
    total += rt
    print(f"location_{loc}: {rt}")
    for c in TEN:
        if counts[loc][c]:
            print(f"    {c:<9} {counts[loc][c]}")
print("TOTAL images:", total)
print("dropped (non-keep class):", dropped, " missing source:", missing)
