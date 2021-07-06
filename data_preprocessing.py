import os
import shutil
import random

random.seed(42)

def split_neu_cls(raw_path, save_path):
    g = os.walk(raw_path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            cls = file_name.split("_")[0].lower()

            part_str = "train" if random.random() > 0.3 else "val"

            src_file = os.path.join(path, file_name)
            dst_path = os.path.join(save_path, part_str, cls)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path, exist_ok=True)
            dst_file = os.path.join(dst_path, file_name)
            shutil.copyfile(src_file, dst_file)
            print(dst_file)

    print("done.")


if __name__ == "__main__":
    raw_path = "data/NEU-CLS/raw"
    save_path = "data/NEU-CLS"
    split_neu_cls(raw_path, save_path)
