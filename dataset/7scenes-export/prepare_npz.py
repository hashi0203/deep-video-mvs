import os

import cv2
import numpy as np
from path import Path
from tqdm import tqdm


def check_data(path):
    image_path = path / "images"
    depth_path = path / "depth"
    image_filenames = sorted(image_path.files("*.png"))
    depth_filenames = sorted(depth_path.files("*.png"))

    frames = []
    for current_index in range(len(image_filenames)):
        image_frame = image_filenames[current_index].split("/")[-1][:-4]
        depth_frame = depth_filenames[current_index].split("/")[-1][:-4]
        assert image_frame == depth_frame, "image and depth files do not match"
        frames.append(image_frame)
    return frames


def save_data(path, frames):
    for frame in tqdm(frames):
        image = cv2.imread(path / "images" / frame + ".png", -1)
        depth = cv2.imread(path / "depth" / frame + ".png", -1)
        np.savez_compressed(os.path.join(path, frame), image=image, depth=depth)


def read_split(path):
    scenes_txt = np.loadtxt(path, dtype=str, delimiter="\n")
    return scenes_txt


if __name__ == '__main__':
    root_folder = Path("/home/nhsmt1123/master-thesis/deep-video-mvs/data/7scenes")
    scenes = np.concatenate([read_split(root_folder / "train.txt"), read_split(root_folder / "validation.txt")])

    for scene in scenes:
        scene_output_path = os.path.join(root_folder, scene)
        if os.path.exists(scene_output_path):
            print('checking image and depth data for %s...' % scene)
            frames = check_data(scene_output_path)
            print('saving image and depth data for %s...' % scene)
            save_data(scene_output_path, frames)
        else:
            print("%s do not exist" % scene_output_path)
            print("execute export color and depth first")