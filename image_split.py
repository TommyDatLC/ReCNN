import os
import csv
import random
import shutil

def split_and_label():
    dataset_dir = "Dataset"  # folder name
    output_dir = "dataset_split"
    split_ratio = [0.7, 0.15, 0.15]

    classes = {"cat": 0, "dog": 1}

    csv_files = {
        "train": open("train_labels.csv", "w", newline=""),
        "val": open("val_labels.csv", "w", newline=""),
        "test": open("test_labels.csv", "w", newline=""),
    }
    writers = {k: csv.writer(v) for k, v in csv_files.items()}
    for w in writers.values():
        w.writerow(["path", "label"])

    for cls, label in classes.items():
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(class_dir):
            print(f" ERROR!!! Folder not found: {class_dir}")
            continue

        print(f"\n Processing {cls} ...")
        for subdir in os.listdir(class_dir):
            subpath = os.path.join(class_dir, subdir)
            if not os.path.isdir(subpath):
                continue

            images = [f for f in os.listdir(subpath)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            random.shuffle(images)
            n_total = len(images)
            n_train = int(n_total * split_ratio[0])
            n_val = int(n_total * split_ratio[1])

            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:]
            }

            for split_name, split_images in splits.items():
                split_path = os.path.join(output_dir, split_name, cls.capitalize(), subdir)
                os.makedirs(split_path, exist_ok=True)

                for img in split_images:
                    src = os.path.join(subpath, img)
                    dst = os.path.join(split_path, img)
                    shutil.copy(src, dst)
                    writers[split_name].writerow([dst, label])

    for f in csv_files.values():
        f.close()

    print("\n Dataset split & labeled successfully!")

if __name__ == "__main__":
    split_and_label()
