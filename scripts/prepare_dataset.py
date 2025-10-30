import os
import argparse
import pandas as pd
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


def load_split_files(raw_dir):
    labels_csv = os.path.join(raw_dir, "Data_Entry_2017.csv")
    trainval_txt = os.path.join(raw_dir, "train_val_list.txt")
    test_txt = os.path.join(raw_dir, "test_list.txt")

    df = pd.read_csv(labels_csv)

    with open(trainval_txt) as f:
        trainval_list = [x.strip() for x in f]
    with open(test_txt) as f:
        test_list = [x.strip() for x in f]

    df["split"] = df["Image Index"].apply(
        lambda x: "test" if x in test_list else "trainval"
    )

    return df, trainval_list, test_list


def assign_train_val(df, val_split):
    trainval_df = df[df["split"] == "trainval"]
    test_df = df[df["split"] == "test"]

    # shuffle for validation split
    trainval_df = trainval_df.sample(frac=1, random_state=42)

    val_count = int(len(trainval_df) * val_split)
    val_df = trainval_df.iloc[:val_count]
    train_df = trainval_df.iloc[val_count:]

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"

    final_df = pd.concat([train_df, val_df, test_df])
    return final_df


def preprocess_and_save(img_path, save_path, img_size=224):
    try:
        img = Image.open(img_path).convert("RGB")

        # center square crop
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        img = img.resize((img_size, img_size))

        img.save(save_path)
        return True
    except Exception:
        return False


def process_row(row, input_dir, output_dir, img_size):
    filename = row["Image Index"]
    split = row["split"]

    # Look for the image inside any images_xxx/images/ folder
    img_path = None
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder, "images", filename)
        if os.path.exists(folder_path):
            img_path = folder_path
            break

    if img_path is None:
        return filename  # missing file

    # create output directories
    dst_dir = os.path.join(output_dir, split)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, filename)

    ok = preprocess_and_save(img_path, dst, img_size)
    if not ok:
        return filename

    return None


def main(args):
    df, trainval_list, test_list = load_split_files(args.input)

    df = assign_train_val(df, args.val_split)

    os.makedirs(args.output, exist_ok=True)

    rows = df.to_dict(orient="records")

    print(f"🔧 Using {args.workers} workers")
    print(f"📊 Total images: {len(rows)}")

    missing = []

    with Pool(args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(
                partial(
                    process_row,
                    input_dir=args.input,
                    output_dir=args.output,
                    img_size=args.img_size,
                ),
                rows,
            ),
            total=len(rows),
            desc="Processing",
        ):
            if result:
                missing.append(result)

    if missing:
        print("\n⚠️ The following files failed or were missing:")
        for m in missing[:20]:
            print(" -", m)
        print(f"Total failures: {len(missing)}")

    print("\n✅ Dataset preparation complete!")
    print(f"Processed dataset at: {args.output}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description="Prepare NIH ChestX-ray14 Dataset")

    parser.add_argument(
        "--input", type=str, required=True, help="Path to raw NIH dataset folder"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save processed dataset"
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Output image size (default 224)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio from train set",
    )
    parser.add_argument(
        "--workers", type=int, default=cpu_count(), help="Number of parallel workers"
    )

    args = parser.parse_args()
    main(args)
