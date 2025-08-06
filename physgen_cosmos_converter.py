import os
import shutil
import re
import cv2
import numpy as np

import prime_printer as prime



def create_or_clear_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"Cleared {folder_path}.")
    else:
        os.makedirs(folder_path)
        print(f"Created {folder_path}.")

def convert_to_cosmos(input_folder, target_folder, output_folder, variation):
    # Path clearing/creating
    output_folder_metas = os.path.join(output_folder, "metas")
    output_folder_videos = os.path.join(output_folder, "videos")
    create_or_clear_folder(folder_path=output_folder)
    create_or_clear_folder(folder_path=output_folder_metas)
    create_or_clear_folder(folder_path=output_folder_videos)

    # Get sorted filenames of input and target images
    input_files = sorted([f for f in os.listdir(input_folder) if f.startswith("input_physgen") and f.endswith(".png")])
    target_files = sorted([f for f in os.listdir(target_folder) if f.startswith("target_physgen") and f.endswith(".png")])

    if len(input_files) != len(target_files):
        raise FileNotFoundError("Amount of input files does not fit to the amount of target files.")

    data_len = len(input_files)
    for idx, (input_file, target_file) in enumerate(zip(input_files, target_files)):
        input_path = os.path.join(input_folder, input_file)
        target_path = os.path.join(target_folder, target_file)
        
        # Load and convert to grayscale
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to the smaller dimensions
        height = min(input_img.shape[0], target_img.shape[0])
        width = min(input_img.shape[1], target_img.shape[1])
        input_img = cv2.resize(input_img, (width, height))
        target_img = cv2.resize(target_img, (width, height))

        # Convert grayscale images to BGR (needed for video encoding)
        input_img_bgr = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        target_img_bgr = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)

        # Initialize video writer
        video_name = input_file.replace("input_", "video_").replace(".png", ".mp4")
        cur_output_path = os.path.join(output_folder_videos, video_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # 30 frames per second
        video_writer = cv2.VideoWriter(cur_output_path, fourcc, fps, (width, height))

        # Alternate: 1s input image, 1s target image -> need 93 frames
        for _ in range(70): # frame 0 - 69 = 70 Frames
            video_writer.write(input_img_bgr)   
        for _ in range(23): # frame 70 - 92 = 23 Frames
            video_writer.write(target_img_bgr)

        video_writer.release()
        # print(f"🎞️ Video saved to: {cur_output_path}")

        # Save Instruction txt
        txt_name = video_name.replace(".mp4", ".txt")
        cur_output_path = os.path.join(output_folder_metas, txt_name)
        with open(cur_output_path, "w") as f:
            f.write(f"Make a {variation} propagation with the sound source in the center of the image in one step.")

        prime.get_progress_bar(total=data_len, progress=idx+1, 
                                should_clear=True, left_bar_char='|', right_bar_char='|', 
                                progress_char='#', empty_char=' ', 
                                front_message='Physgen Data Convertion', back_message='', size=15)
    print(f"\nSuccessfull saved {data_len} datapoints into {os.path.abspath(output_folder)}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Physgen Dataset into Cosmos Format (video + text instructions).")

    parser.add_argument("--input_folder", type=str, required=True, help="Path to existing OSM input images")
    parser.add_argument("--target_folder", type=str, required=True, help="Path to existing real target images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the target output folder, will have path/videos and path/metas subfolders")
    parser.add_argument("--variation", type=str, required=True, help="PhysGen variation (e.g. box_texture, box_position, etc.)")

    args = parser.parse_args()

    convert_to_cosmos(
            input_folder=args.input_folder, 
            target_folder=args.target_folder, 
            output_folder=args.output_folder,
            variation=args.variation
    )
    
