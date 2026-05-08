import os, shutil, random
from pathlib import Path

SOURCE = "animals/animals"
DEST   = "data"
SPLIT  = 0.8

classes = os.listdir(SOURCE)
print(f"Found classes: {classes}")

for cls in classes:
    src = Path(SOURCE) / cls
    images = list(src.glob("*.*"))
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest_dir = Path(DEST) / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, dest_dir / img.name)

    print(f"  {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

print("\nDone! Dataset ready.")