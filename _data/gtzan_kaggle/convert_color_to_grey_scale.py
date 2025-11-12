import os
from PIL import Image
import numpy as np

# ==============================
# Settings
# ==============================
src_root = r"C:\Users\natha\OneDrive\Desktop\0_Fall_25\0_510\0_project\gtzan_kaggle\Data\images_original"
dst_root = r"C:\Users\natha\OneDrive\Desktop\0_Fall_25\0_510\0_project\gtzan_kaggle\Data\images_grey_scale"

target_size = (128, 128)   # (width, height)
normalize = True           # scale to [0, 1]
report_every = 100

# ==============================
# Conversion loop
# ==============================
os.makedirs(dst_root, exist_ok=True)
genres = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
print(f"Found genres: {genres}")

all_mins, all_maxs = [], []

for genre in genres:
    src_dir = os.path.join(src_root, genre)
    dst_dir = os.path.join(dst_root, genre)
    os.makedirs(dst_dir, exist_ok=True)

    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n🎶 Processing {genre}: {len(images)} images")

    for i, img_name in enumerate(images, 1):
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)

        try:
            with Image.open(src_path) as img:
                gray = img.convert("L")

                # Check or resize
                if gray.size != target_size:
                    gray = gray.resize(target_size)

                arr = np.array(gray, dtype=np.float32)

                # Normalize pixel values to [0, 1]
                if normalize:
                    arr = arr / 255.0
                    all_mins.append(arr.min())
                    all_maxs.append(arr.max())

                # Convert back to image and save (rescale to 0–255 for PNG)
                out_img = Image.fromarray((arr * 255).astype(np.uint8))
                out_img.save(dst_path)

        except Exception as e:
            print(f"❌ Error on {img_name}: {e}")

        if i % report_every == 0:
            print(f"  → {i} images processed in {genre}")

# ==============================
# Report stats
# ==============================
if normalize and all_mins and all_maxs:
    print("\n📊 Normalization check:")
    print(f"  Global min: {np.min(all_mins):.4f}")
    print(f"  Global max: {np.max(all_maxs):.4f}")
else:
    print("\n(no normalization stats collected)")

print("\n✅ Conversion + normalization complete!")
print(f"Grey-scale images saved to:\n{dst_root}")
