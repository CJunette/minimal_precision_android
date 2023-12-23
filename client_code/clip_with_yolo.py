import os
from multiprocessing import Pool
import cv2
from torchvision.transforms import ToPILImage
import numpy as np
import torch
from pathlib import Path
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import configs
import sys


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


def process_images(model, device, image_path_list, output_path_list, log_file):
    images = [Image.open(p) for p in image_path_list]
    images_tensor = [torchvision.transforms.ToTensor()(img).unsqueeze(0) for img in images]  # Convert to tensor
    images_tensor = [img_tensor.to(device) for img_tensor in images_tensor]  # Move tensors to GPU if available
    images_batch = torch.cat(images_tensor, dim=0)

    import matplotlib
    matplotlib.use('TkAgg')
    results = model(images)
    results_pandas = results.pandas()
    selected_result_list = [None for _ in range(len(results_pandas.xyxy))]
    for index, result in enumerate(results_pandas.xyxy):
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
            print(f"No person detected in image {image_path_list[index]}")
            log_file.write(f"No person detected in image {image_path_list[index]}")

    to_pil = ToPILImage()
    cropped_images = [None for _ in range(len(results_pandas.xyxy))]

    for index in range(len(selected_result_list)):
        x1, y1, x2, y2 = selected_result_list[index][0], selected_result_list[index][1], selected_result_list[index][2], selected_result_list[index][3]
        center_x = (x1 + x2) / 2
        half_width = configs.clipped_image_width // 2
        cropped_x1 = (center_x - half_width).astype(int)
        cropped_x2 = (center_x + half_width).astype(int)
        cropped_y1 = (y1 - 25).astype(int)
        cropped_y2 = (cropped_y1 + half_width * 2).astype(int)

        # # 检查裁剪框是否正确。
        # from matplotlib import patches
        # fig, ax = plt.subplots(1)
        # ax.imshow(images[0])
        # rect = patches.Rectangle((cropped_x1, cropped_y1), cropped_x2-cropped_x1, cropped_y2-cropped_y1, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        if cropped_x1 < 0 or cropped_y1 < 0 or cropped_x2 > images_tensor[index].shape[3] or cropped_y2 > images_tensor[index].shape[2]:
            print(f"Image {image_path_list[index]} is out of bounds.")
            log_file.write(f"Image {image_path_list[index]} is out of bounds.")

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
        cropped_images[index] = cropped_image

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
    root_dir = f'output/subject_{subject_name}/clipped_{configs.mode}'
    file_path_list = os.listdir(root_dir)
    file_path_list.sort(key=get_row_and_col)

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
        log_file = open(f"{log_path_prefix}/log_{configs.subject_num}.txt", "w")
    else:
        log_file = open(f"{log_path_prefix}/log_{subject_num}.txt", "w")

    image_name_list_1 = []
    image_path_list_1 = []
    if subject_num is None:
        file_path_1 = f"output/subject_{configs.subject_num}/"
    else:
        file_path_1 = f"output/subject_{subject_num}/"

    file_names_1 = os.listdir(file_path_1)
    for file_name_1 in file_names_1:
        if file_name_1.startswith("row_"):
            image_name_list_2 = []
            image_path_list_2 = []
            file_path_2 = f"{file_path_1}{file_name_1}/"
            file_names_2 = os.listdir(file_path_2)
            for file_name_2 in file_names_2:
                file_name_3 = f"{file_path_2}{file_name_2}"
                image_name_list_2.append(file_name_3)
                file_path_3 = f"{file_path_2}".replace(f"{configs.mode}", f"clipped_{configs.mode}")
                image_path_list_2.append(file_path_3)

            image_name_list_1.append(np.array(image_name_list_2))
            image_path_list_1.append(np.array(image_path_list_2))

    image_name_list_1 = np.array(image_name_list_1)
    image_path_list_1 = np.array(image_path_list_1)

    image_name_list_1d = image_name_list_1.reshape(-1)
    image_path_list_1d = image_path_list_1.reshape(-1)

    repeat_times = 200
    single_time_amount = len(image_name_list_1d) // repeat_times
    for i in range(0, repeat_times):
        print(f"repeat package {i}")
        log_file.write(f"repeat package {i}\n")
        process_images(model, device, image_name_list_1d[i * single_time_amount: (i + 1) * single_time_amount], image_path_list_1d[i * single_time_amount: (i + 1) * single_time_amount], log_file)

    log_file.close()

