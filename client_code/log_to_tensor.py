import multiprocessing
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


def transform_single_image_to_tensor(df, row_index, subject_name, specific_path):
    image_path = df.iloc[row_index]["image_path"]
    mode_str = image_path.split(f"{subject_name}/")[1].split("/")[0]
    save_name = image_path.replace("output", "output_tensor").replace(mode_str, specific_path).replace(".jpg", ".pt")
    save_path = save_name.split("/capture_")[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    row = df.iloc[row_index]
    location_list = []
    for column in df.columns:
        if column == "image_path" or column == "other":
            continue
        location_list.append(row[column])
    tensor_list = torch.tensor(location_list, dtype=torch.float32)
    torch.save(tensor_list, save_name)


def log_to_tensor(specific_path: str):
    file_prefix = "log"
    subject_names = os.listdir(file_prefix)
    subject_file_list = []
    for subject_name in subject_names:
        if subject_name.startswith(f"subject_"):
            subject_file_list.append(f"{file_prefix}/{subject_name}/{specific_path}.csv")
    # 根据subject_file_list中的文件夹名字，通过他们的subject_后的编号进行排序
    subject_file_list.sort(key=lambda x: float(x.replace(f"{file_prefix}/subject_", "").replace(f"/{specific_path}.csv", "")))

    for subject_index, subject_file_name in enumerate(subject_file_list):
        arg_list = []

        df = pd.read_csv(f"{subject_file_name}")
        for i in range(df.shape[0]):
            arg_list.append((df, i, subject_names[subject_index], specific_path))

        with multiprocessing.Pool(configs.num_of_process) as p:
            p.starmap(transform_single_image_to_tensor, arg_list)



