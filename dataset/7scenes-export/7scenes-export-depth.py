import shutil

import cv2
import numpy as np
from path import Path

scenes = [("chess", "01", "02"),
          ("fire", "01", "02"),
          ("heads", "02"),
          ("office", "01", "03"),
          ("pumpkin", "03", "06"),
          ("redkitchen", "01", "07"),
          ("stairs", "02", "06"), # train
          ("chess", "03"),
          ("fire", "03", "04"),
          ("heads", "01"),
          ("office", "02"),
          ("pumpkin", "01"),
          ("redkitchen", "03"),
          ("stairs", "01")] # test

input_folder = Path("/home/share/dataset/7scenes")
output_folder = Path("/home/nhsmt1123/master-thesis/deep-video-mvs/data/7scenes")
for scene in scenes:

    if len(scene) == 3:
        folder_name, seq1, seq2 = scene
        seqs = [seq1, seq2]
    else:
        folder_name, seq1 = scene
        seqs = [seq1]

    scene_input_folder = input_folder / folder_name

    for seq in seqs:
        files = sorted((scene_input_folder / "seq-" + seq).files("*depth.png"))

        room_name = folder_name.split("_")[-1]
        scene_name = room_name + "-seq-" + seq
        scene_output_folder = output_folder / scene_name / 'depth'
        if scene_output_folder.exists():
            shutil.rmtree(scene_output_folder)
        scene_output_folder.mkdir()
        for index, file in enumerate(files):
            depth = cv2.imread(file, -1)
            depth_uint = np.round(depth).astype(np.uint16)
            save_filename = scene_output_folder / (str(index).zfill(6) + ".png")
            cv2.imwrite(save_filename, depth_uint, [cv2.IMWRITE_PNG_COMPRESSION, 3])
