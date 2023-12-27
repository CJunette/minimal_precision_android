import os
from multiprocessing import Pool
import cv2
import pandas as pd
from torchvision.transforms import ToPILImage
import numpy as np
import torch
from pathlib import Path
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import configs
import sys
import util_functions
import matplotlib
from matplotlib import patches


# original_sys_path = sys.path.copy()
# sys.path.append('./models/yolov5')
# from models.yolo import Model
# sys.path = original_sys_path


def crop_image_with_torch(img_tensor, center_x, center_y, half_width):
    """
    Crop an image tensor using PyTorch.
    """
    # Calculate the cropping coordinates
    _, c, h, w = img_tensor.shape
    top = int(center_y - half_width)
    left = int(center_x - half_width)
    bottom = int(center_y + half_width)
    right = int(center_x + half_width)

    # Ensure coordinates are within image boundaries
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    return img_tensor[:, :, top:bottom, left:right]


def crop_image_single_pool(img_tensor, pred, image_path, image, output_path):
    print(image_path)
    # Filter predictions for the 'person' class with confidence > 0.8
    person_preds = [x for x in pred if x[4] > 0.8 and int(x[5]) == 0]  # '0' is the class index for 'person' in COCO dataset
    if not person_preds:
        print(f"No person detected in image {image_path}")
        return

    box = person_preds[0][:4]  # Get the highest confidence prediction
    head_shoulder_height = (box[3] - box[1]) * 0.5
    center_x = (box[0] + box[2]) / 2
    center_y = box[1] - head_shoulder_height

    half_width = 150
    cropped_image = image.crop((center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width))

    # cropped_tensor = crop_image_with_torch(img_tensor, center_x, center_y, half_width)
    # a = cropped_tensor.cpu().numpy()
    # cropped_image = Image.fromarray((cropped_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    # plt.imshow(cropped_image)
    # plt.show()

    # output_path = output_path
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # cropped_image.save(f"{output_path}capture{image_path.split('capture')[1]}")

    return cropped_image


def get_interpolate_result(selected_result_list, current_index, previous_index, next_index):
    previous_result = np.array(selected_result_list[previous_index])
    next_result = np.array(selected_result_list[next_index])
    current_result = previous_result + (next_result - previous_result) * (current_index - previous_index) / (next_index - previous_index)
    current_result = current_result.tolist()
    return current_result


def process_images(model, device, image_path_list, output_path_list, log_file, log_file_path):
    images = [Image.open(p) for p in image_path_list]
    images_tensor = [torchvision.transforms.ToTensor()(img).unsqueeze(0) for img in images]  # Convert to tensor
    images_tensor = [img_tensor.to(device) for img_tensor in images_tensor]  # Move tensors to GPU if available
    images_batch = torch.cat(images_tensor, dim=0)
    # matplotlib.use('TkAgg')

    results = model(images)
    results_pandas = results.pandas()
    selected_result_list = [None for _ in range(len(results_pandas.xyxy))]
    for index, result in enumerate(results_pandas.xyxy):
        # check if image_path[index] is in log_file(pd.dataframe), if not, add it.
        if image_path_list[index] not in log_file["image_path"].values:
            log_file.loc[log_file.shape[0]] = [image_path_list[index], "", "", "", "", ""]

        selected_result = result[(result["confidence"] > 0.7) & (result["name"] == "person")]
        if selected_result.shape[0] > 0:
            selected_result = selected_result.sort_values(by="confidence", ascending=False)
            selected_result_list[index] = [selected_result["xmin"].iloc[0], selected_result["ymin"].iloc[0], selected_result["xmax"].iloc[0], selected_result["ymax"].iloc[0]]

            # 检查框选是否正确。
            # fig, ax = plt.subplots(1)
            # ax.imshow(images[0])
            # print(f"{selected_result_list[index][0]:.3f}, {selected_result_list[index][1]:.3f}, {selected_result_list[index][2]:.3f}, {selected_result_list[index][3]:.3f}")
            # x1, y1, x2, y2 = selected_result_list[index][0], selected_result_list[index][1], selected_result_list[index][2], selected_result_list[index][3]
            # from matplotlib import patches
            # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # plt.show()
        else:
            log_file.loc[log_file["image_path"] == image_path_list[index], "other"] = "No person detected"
            print(f"No person detected in image {image_path_list[index]}")
            # log_file.write(f"No person detected in image {image_path_list[index]}")

    # if none exist in selected_result_list, interpolate the result.
    selected_result_list = util_functions.interpolate_none(selected_result_list, get_interpolate_result, None) # 下方为源代码，我将源代码稍微修改了一下然后集成到util_functions.py中。
    # check_none_index = 0
    # while check_none_index < len(selected_result_list):
    #     if selected_result_list[check_none_index] is None:
    #         if check_none_index == 0:
    #             next_index = 1
    #             while next_index < len(selected_result_list):
    #                 if selected_result_list[next_index] is not None:
    #                     for i in range(check_none_index, next_index):
    #                         selected_result_list[i] = selected_result_list[next_index]
    #                     break
    #                 next_index += 1
    #             check_none_index = next_index
    #         elif check_none_index == len(selected_result_list) - 1:
    #             selected_result_list[check_none_index] = selected_result_list[check_none_index - 1]
    #         else:
    #             next_index = check_none_index + 1
    #             previous_index = check_none_index - 1
    #             while next_index <= len(selected_result_list):
    #                 if next_index < len(selected_result_list) and selected_result_list[next_index] is not None:
    #                     for i in range(previous_index + 1, next_index):
    #                         selected_result_list[i] = get_interpolate_result(selected_result_list, i, previous_index, next_index)
    #                     break
    #                 elif next_index == len(selected_result_list):
    #                     for i in range(check_none_index, next_index):
    #                         selected_result_list[i] = selected_result_list[previous_index]
    #                     break
    #                 else:
    #                     next_index += 1
    #             check_none_index = next_index
    #     else:
    #         check_none_index += 1

    to_pil = ToPILImage()
    cropped_images = [None for _ in range(len(results_pandas.xyxy))]

    for index in range(len(selected_result_list)):
        x1, y1, x2, y2 = selected_result_list[index][0], selected_result_list[index][1], selected_result_list[index][2], selected_result_list[index][3]
        center_x = (x1 + x2) / 2
        half_width = configs.clipped_image_width // 2
        cropped_x1 = int(center_x - half_width)
        cropped_x2 = int(center_x + half_width)
        center_y = (y1 + y2) / 2 + 200
        cropped_y1 = int(center_y - half_width)
        cropped_y2 = int(center_y + half_width)

        log_file.loc[log_file["image_path"] == image_path_list[index], "start_x"] = cropped_x1
        log_file.loc[log_file["image_path"] == image_path_list[index], "start_y"] = cropped_y1
        log_file.loc[log_file["image_path"] == image_path_list[index], "end_x"] = cropped_x2
        log_file.loc[log_file["image_path"] == image_path_list[index], "end_y"] = cropped_y2

        # # 检查裁剪框是否正确。
        # fig, ax = plt.subplots(1)
        # ax.imshow(images[0])
        # rect = patches.Rectangle((cropped_x1, cropped_y1), cropped_x2-cropped_x1, cropped_y2-cropped_y1, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        if cropped_x1 < 0 or cropped_y1 < 0 or cropped_x2 > images_tensor[index].shape[3] or cropped_y2 > images_tensor[index].shape[2]:
            # print(f"Image {image_path_list[index]} is out of bounds.")
            # log_file.write(f"Image {image_path_list[index]} is out of bounds.")

            cropped_tensor = torch.zeros(3, half_width * 2, half_width * 2).to(images_tensor[index].device)
            y1, y2 = max(cropped_y1, 0), min(cropped_y2, images_tensor[index].shape[2])
            x1, x2 = max(cropped_x1, 0), min(cropped_x2, images_tensor[index].shape[3])
            start_y = max(0, -cropped_y1)
            start_x = max(0, -cropped_x1)

            end_y = start_y + (y2 - y1)
            end_x = start_x + (x2 - x1)
            cropped_tensor[:, start_y:end_y, start_x:end_x] = images_tensor[index][0, :, y1:y2, x1:x2]
            cropped = cropped_tensor

        else:
            cropped = images_tensor[index][0, :, cropped_y1:cropped_y2, cropped_x1:cropped_x2]

        cropped_image = to_pil(cropped.cpu())
        cropped_image = cropped_image.resize((configs.resized_image_width, configs.resized_image_width))
        cropped_images[index] = cropped_image

    log_file.to_csv(f"{log_file_path}", index=False)

    for i in range(len(cropped_images)):
        # plt.imshow(cropped_images[i])
        # plt.show()
        output_path = output_path_list[i]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cropped_images[i].save(f"{output_path}capture{image_path_list[i].split('capture')[1]}")


def read_model_and_device():
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    # os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'

    # model = Model(cfg='models/yolov5/models/yolov5s.yaml')
    # weights = torch.load("models/yolov5/yolov5_weight/yolov5s.pt")
    # model.load_state_dict(weights)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = torch.load("models/yolov5/yolov5_weight/yolov5s.pt", map_location=torch.device(device))
    model = torch.hub.load("models/yolov5", "custom", "models/yolov5/yolov5_weight/yolov5l.pt", source="local", force_reload=True)
    # model = torch.hub.load("models/yolov5", "custom", "models/yolov5/yolov5_weight/yolov5m.pt", source="local")

    model.to(device)
    model.eval()

    return model, device


def resize_clipped_images(subject_name: str, resize_width: int):
    '''
    本次没有用到这个函数。
    :param subject_name:
    :param resize_width:
    :return:
    '''
    root_dir = f'output/subject_{subject_name}/clipped_{configs.mode}'
    file_path_list = os.listdir(root_dir)
    file_path_list.sort(key=util_functions.get_row_and_col)

    for col_row_file_name in file_path_list:
        file_path = f"{root_dir}/{col_row_file_name}"
        image_names = os.listdir(file_path)
        image_names.sort(key=lambda x: int(x.replace(".jpg", "").replace("capture_", "")))

        for image_name in image_names:
            image_path = f"{file_path}/{image_name}"
            image = Image.open(image_path)
            image = image.resize((resize_width, resize_width))

            # show image
            # plt.imshow(image)
            # plt.show()
            save_path = f"{file_path}".replace(f"clipped_{configs.mode}", f"clipped_{configs.mode}_{resize_width}/")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = f"{save_path}/{image_name}"
            image.save(save_name)


def clip_human_in_batch(subject_num=None):
    model, device = read_model_and_device()

    # prepare log file
    log_path_prefix = "log"
    if not os.path.exists(log_path_prefix):
        os.makedirs(log_path_prefix)

    if subject_num is None:
        log_file_path = f"{log_path_prefix}/subject_{configs.subject_num}"
    else:
        log_file_path = f"{log_path_prefix}/subject_{subject_num}"

    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
        log_file_name = f"{log_file_path}/log_clip.csv"
        log_file_header = ["image_path", "start_x", "start_y", "end_x", "end_y", "other"]
        log_file = pd.DataFrame(columns=log_file_header)
        log_file.to_csv(log_file_name, index=False)

    image_name_list_1 = []
    image_path_list_1 = []
    if subject_num is None:
        file_path_1 = f"output/subject_{configs.subject_num}/camera_distant"
    else:
        file_path_1 = f"output/subject_{subject_num}/camera_distant"

    file_names_1 = os.listdir(file_path_1)
    file_names_1.sort(key=util_functions.get_row_and_col)

    for file_name_1 in file_names_1:
        if file_name_1.startswith("row_"):
            image_name_list_2 = []
            image_path_list_2 = []
            file_path_2 = f"{file_path_1}/{file_name_1}/"
            file_names_2 = os.listdir(file_path_2)
            file_names_2.sort(key=lambda x: int(x.replace(".jpg", "").replace("capture_", "")))
            for file_index_2, file_name_2 in enumerate(file_names_2):
                # if file_index_2 >= 20: # 本来想只取前20个的，但发现有些数据样本没采满20个。
                #     continue
                file_name_3 = f"{file_path_2}{file_name_2}"
                image_name_list_2.append(file_name_3)
                file_path_3 = f"{file_path_2}".replace(f"camera_distant", f"clipped_camera_distant")
                image_path_list_2.append(file_path_3)

            image_name_list_1.append(np.array(image_name_list_2))
            image_path_list_1.append(np.array(image_path_list_2))

    for i in range(0, len(image_name_list_1)):
        log_file = pd.read_csv(log_file_path)
        process_images(model, device, image_name_list_1[i], image_path_list_1[i], log_file, log_file_path)


