import os
from multiprocessing import Pool, Semaphore
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage
import configs
import util_functions


def transform_single_image_to_tensor(prefix_path, image_name, transform):
    max_open_files = Semaphore(100)  # 例如，同时只允许100个文件打开

    with max_open_files:
        with Image.open(f"{prefix_path}/{image_name}.jpg") as image:
            image = image.convert("RGB")
            image_tensor = transform(image)
            image_tensor = (image_tensor * 255).type(torch.uint8) # 需要转化为uint8，读取速度与文件大小有关。
            save_path = prefix_path.replace("output", "output_tensor")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(image_tensor, f"{save_path}/{image_name}.pt")


def image_to_tensor(specific_path: str, resized_size: int=0):
    file_prefix = "output/"
    subject_file_names = os.listdir(file_prefix)
    subject_file_list = []
    for subject_file_name in subject_file_names:
        if subject_file_name.startswith("subject_"):
            subject_file_list.append(f"{file_prefix}{subject_file_name}/{specific_path}")
    # 根据subject_file_list中的文件夹名字，通过他们的subject_后的编号进行排序
    subject_file_list.sort(key=lambda x: float(x.replace("output/subject_", "").replace(f"/{specific_path}", "")))

    if resized_size > 0:
        transform = transforms.Compose([
            transforms.Resize((resized_size, resized_size)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    for subject_index, subject_file_path in enumerate(subject_file_list):
        print(f"subject_index: {subject_index}")
        col_row_file_names = os.listdir(subject_file_path)
        col_row_file_names.sort(key=util_functions.get_row_and_col)

        arg_list = []
        for file_index, col_row_file_name in enumerate(col_row_file_names):
            image_names = os.listdir(f"{subject_file_path}/{col_row_file_name}")
            image_names.sort(key=lambda x: int(x.replace(".jpg", "").replace("capture_", "")))

            # 顺便检查一下每个文件夹中的图片数量是否为10。
            # jpg_names = []
            # for image_name in image_names:
            #     if image_name.endswith(".jpg"):
            #         jpg_names.append(image_name)
            # if len(jpg_names) < 10:
            #     print(f"subject_index: {subject_index}, file_index: {col_row_file_name}")

            for image_name in image_names:
                if image_name.endswith(".jpg"):
                    arg_list.append([f"{subject_file_path}/{col_row_file_name}", f"{image_name.replace('.jpg', '')}", transform])
        with Pool(configs.num_of_process) as p:
            p.starmap(transform_single_image_to_tensor, arg_list)
            p.close()
            p.join()


