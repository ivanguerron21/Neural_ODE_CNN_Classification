import os, glob, random
import shutil
from pathlib import Path
import splitfolders


def count_images(folder):
    for car in os.listdir(folder):
        m = len(os.listdir(f"{folder}/{car}"))
        print(car, m)


def generate_even_dataset(src_dir, dst_dir, num):
    if not Path(dst_dir).exists():
        Path(dst_dir).mkdir()
    for car in os.listdir(src_dir):
        file_path_type = f'{src_dir}/{car}/*.jpg'
        images = random.sample(glob.glob(file_path_type), num)
        if not Path(f'{dst_dir}/{car}').exists():
            Path(f'{dst_dir}/{car}').mkdir()
        for i in images:
            shutil.copy(i, i.replace(src_dir, dst_dir))


if __name__ == '__main__':
    src_dir = "105_classes_pins_dataset"
    dst_dir = "new_dataset"
    final_data = "split_data"
    generate_even_dataset(src_dir, dst_dir, 86)
    splitfolders.ratio(dst_dir, final_data, seed=33, ratio=(.7, .2, .1), group_prefix=None, move=False)



