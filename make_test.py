import os, shutil, random
from pathlib import Path

random.seed(42)
val  = Path('data/val')
test = Path('data/test')

for cls_folder in val.iterdir():
    if not cls_folder.is_dir():
        continue
    images = list(cls_folder.glob('*.*'))
    test_imgs = random.sample(images, max(1, int(len(images) * 0.5)))
    dest = test / cls_folder.name
    dest.mkdir(parents=True, exist_ok=True)
    for img in test_imgs:
        shutil.copy(img, dest / img.name)
    print(f"  {cls_folder.name}: {len(test_imgs)} images")

print('Done! Test set created.')