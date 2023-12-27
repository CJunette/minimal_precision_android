import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from matplotlib.colors import LinearSegmentedColormap

import configs
import regression_with_resnet
import util_functions


def manual_heatmap(avg_accuracy_list_1, colors, subject_index: int, mode: str, col_row_names):
    # 手动绘制热力图。
    fig, ax = plt.subplots(figsize=(24, 40))

    max_avg_accuracy = np.max(avg_accuracy_list_1)
    min_avg_accuracy = np.min(avg_accuracy_list_1)

    cmap = LinearSegmentedColormap.from_list(name='custom', colors=colors, N=100)

    for col_row_index in range(len(col_row_names)):
        col_index = float(col_row_names[col_row_index].split("-")[1].replace("col_", ""))
        row_index = float(col_row_names[col_row_index].split("-")[0].replace("row_", ""))
        if row_index == 12:
            print()
        color_percentage = (avg_accuracy_list_1[col_row_index] - min_avg_accuracy) / (max_avg_accuracy - min_avg_accuracy)
        color = cmap(color_percentage)
        bright = 0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[2]

        if 12 <= row_index <= 15:
            width = 1/3
        else:
            width = 1

        rect = patches.Rectangle((col_index - width/2, configs.row_num - 1 - row_index - width/2), width, width, color=color)
        ax.add_patch(rect)
        # add text
        text_color = "black" if bright > 0.5 else "white"
        text = ax.text(col_index, configs.row_num - 1 - row_index, f"{avg_accuracy_list_1[col_row_index]:.2f}", ha="center", va="center", color=text_color, fontsize=6)
        ax.add_artist(text)

    ax.set_xlim(-0.25*configs.col_num, 1.25*configs.col_num)
    ax.set_ylim(-0.25*configs.row_num, 1.25*configs.row_num)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_avg_accuracy, vmax=max_avg_accuracy))
    cbar = plt.colorbar(sm, ax=ax)

    save_prefix = f"pic/error_heatmap/normalized_values"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    plt.savefig(f"{save_prefix}/{subject_index}_{mode}.jpeg", dpi=300)


def manual_prediction_error(avg_accuracy_list_1, prediction_list_1, colors, subject_index: int, mode: str, col_row_names):
    # 手动绘制误差位置图。
    fig, ax = plt.subplots(figsize=(18, 30))

    max_avg_accuracy = np.max(avg_accuracy_list_1)
    min_avg_accuracy = np.min(avg_accuracy_list_1)

    cmap = LinearSegmentedColormap.from_list(name='custom', colors=colors, N=100)
    for col_row_index in range(len(col_row_names)):
        col_index = float(col_row_names[col_row_index].split("-")[1].replace("col_", ""))
        row_index = float(col_row_names[col_row_index].split("-")[0].replace("row_", ""))

        color_percentage = (avg_accuracy_list_1[col_row_index] - min_avg_accuracy) / (max_avg_accuracy - min_avg_accuracy)
        color = cmap(color_percentage)
        bright = 0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[2]

        if 12 <= row_index <= 15:
            width = 1/3
        else:
            width = 1

        rect = patches.Rectangle((col_index - width/2, configs.row_num - 1 - row_index - width/2), width, width, color=color)
        ax.add_patch(rect)

        print(row_index, col_index)
        # 将错误预测位置与正确位置之间的连线画出来。
        for prediction in prediction_list_1[col_row_index]:
            ax.plot([prediction[1], col_index], [configs.row_num - 1 - prediction[0], configs.row_num - 1 - row_index], color="black", linewidth=0.5, alpha=0.2)
            ax.scatter(prediction[1], configs.row_num - 1 - prediction[0], color=color, s=1, edgecolors="red", zorder=10)

    ax.set_xlim(-0.25*configs.col_num, 1.25*configs.col_num)
    ax.set_ylim(-0.25*configs.row_num, 1.25*configs.row_num)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_avg_accuracy, vmax=max_avg_accuracy))
    cbar = plt.colorbar(sm, ax=ax)

    save_prefix = f"pic/error_heatmap/normalized"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    plt.savefig(f"{save_prefix}/{subject_index}_{mode}.jpeg", dpi=300)


def read_tensor_path_of_row_and_col(validation_prefix, col_row_name, bool_data_used):
    if bool_data_used:
        tensor_prefix = f"{validation_prefix}/{col_row_name}"
        tensor_paths = os.listdir(tensor_prefix)
        tensor_paths.sort(key=lambda x: int(x.split(".")[0].replace("capture_", "")))
        tensor_paths = [f"{tensor_prefix}/{tensor_path}" for tensor_path in tensor_paths]
        return tensor_paths
    else:
        return []


def visualize_error_over_the_map(subject_index: int, model_file_name: str, mode: str = "all_data"):
    util_functions.set_configs_bool_camera_given_mode(mode)

    model_path = f"models/custom_resnet_regression/subject_{str(subject_index).zfill(3)}_{mode}/{model_file_name}"
    # load model to cuda 0
    model = regression_with_resnet.SimpleResNetLightning.load_from_checkpoint(model_path)
    model.cuda(0)
    model.eval()

    clipped_camera_distant_validation_prefix = f"output_tensor/subject_{subject_index}.1/clipped_camera_distant"
    clipped_camera_left_eye_validation_prefix = f"output_tensor/subject_{subject_index}.1/clipped_camera_left_eye"
    clipped_camera_right_eye_validation_prefix = f"output_tensor/subject_{subject_index}.1/clipped_camera_right_eye"
    camera_left_eye_validation_prefix = f"output_tensor/subject_{subject_index}.1/camera_left_eye"
    camera_right_eye_validation_prefix = f"output_tensor/subject_{subject_index}.1/camera_right_eye"
    log_clip_validation_prefix = f"output_tensor/subject_{subject_index}.1/log_clip"
    log_clip_eye_validation_prefix = f"output_tensor/subject_{subject_index}.1/log_clip_eye"

    col_row_names = os.listdir(clipped_camera_distant_validation_prefix)
    col_row_names.sort(key=util_functions.get_row_and_col)

    prediction_list_1 = []
    accuracy_list_1 = []
    for col_row_index, col_row_name in enumerate(col_row_names):
        print(col_row_index)

        # 根据configs中不同camera的bool值，来判断对应的tensor是否应该被读取。
        clipped_camera_distant_tensor_paths = read_tensor_path_of_row_and_col(clipped_camera_distant_validation_prefix, col_row_name, configs.bool_clipped_distant_camera)
        clipped_camera_left_eye_tensor_paths = read_tensor_path_of_row_and_col(clipped_camera_left_eye_validation_prefix, col_row_name, configs.bool_clipped_distant_left_eye_camera)
        clipped_camera_right_eye_tensor_paths = read_tensor_path_of_row_and_col(clipped_camera_right_eye_validation_prefix, col_row_name, configs.bool_clipped_distant_right_eye_camera)
        camera_left_eye_tensor_paths = read_tensor_path_of_row_and_col(camera_left_eye_validation_prefix, col_row_name, configs.bool_left_eye_camera)
        camera_right_eye_tensor_paths = read_tensor_path_of_row_and_col(camera_right_eye_validation_prefix, col_row_name, configs.bool_right_eye_camera)
        # 如同在regression_with_resnet中一样，对tensor paths的长度进行筛选，取最小的长度。
        clipped_distant_file_num = len(clipped_camera_distant_tensor_paths)
        clipped_distant_left_eye_file_num = len(clipped_camera_left_eye_tensor_paths)
        clipped_distant_right_eye_file_num = len(clipped_camera_right_eye_tensor_paths)
        clipped_left_eye_file_num = len(camera_left_eye_tensor_paths)
        clipped_right_eye_file_num = len(camera_right_eye_tensor_paths)

        file_num_list = [clipped_distant_file_num,
                         clipped_distant_left_eye_file_num, clipped_distant_right_eye_file_num,
                         clipped_left_eye_file_num, clipped_right_eye_file_num]
        bool_list = [configs.bool_clipped_distant_camera,
                     configs.bool_clipped_distant_left_eye_camera, configs.bool_clipped_distant_right_eye_camera,
                     configs.bool_left_eye_camera, configs.bool_right_eye_camera]
        tensor_path_list = [clipped_camera_distant_tensor_paths,
                            clipped_camera_left_eye_tensor_paths, clipped_camera_right_eye_tensor_paths,
                            camera_left_eye_tensor_paths, camera_right_eye_tensor_paths]
        length_list = []
        selected_tensor_path_list = []
        for bool_index, bool_value in enumerate(bool_list):
            if bool_value == 1:
                length_list.append(file_num_list[bool_index])
                selected_tensor_path_list.append(tensor_path_list[bool_index])

        minimal_length = max(min(length_list), configs.tensor_select_end_index)

        row_index = float(col_row_name.split("-")[0].replace("row_", ""))
        col_index = float(col_row_name.split("-")[1].replace("col_", ""))

        prediction_list_2 = []
        accuracy_list_2 = []

        # 遍历每个tensor文件，将其读取出来，然后送入模型中进行预测。
        for tensor_file_index in range(5, minimal_length):
            image_tensor_list = []
            for tensor_path_list_index, tensor_path_list_value in enumerate(selected_tensor_path_list):
                tensor_path = tensor_path_list_value[tensor_file_index]
                tensor = torch.load(tensor_path).type(torch.float32) / 255
                tensor = tensor.unsqueeze(0).cuda(0)
                image_tensor_list.append(tensor)

            location_list = []
            if configs.bool_clipped_distant_camera:
                path = f"{log_clip_validation_prefix}/{col_row_name}/capture_{tensor_file_index}.pt"
                location = torch.load(path)
                location_list.append(location)
            if configs.bool_clipped_distant_left_eye_camera:
                path = f"{log_clip_eye_validation_prefix}/{col_row_name}/capture_{tensor_file_index}.pt"
                location = torch.load(path)
                location_list.append(location)
            if location_list:
                location_tensor = torch.concat(location_list, dim=0)
            else:
                location_tensor = torch.tensor([], dtype=torch.float32)
            location_tensor = location_tensor.unsqueeze(0).cuda(0)

            prediction = model(image_tensor_list, location_tensor).squeeze(0).cpu().detach().numpy()
            prediction_x = prediction[0] * (configs.row_num - 1)
            prediction_y = prediction[1] * (configs.col_num - 1)
            prediction_list_2.append([prediction_x, prediction_y])

            accuracy = math.sqrt((prediction_x - row_index) ** 2 + (prediction_y - col_index) ** 2)
            accuracy_list_2.append(accuracy)

        accuracy_list_1.append(accuracy_list_2)
        prediction_list_1.append(prediction_list_2)

    avg_accuracy_list_1 = [sum(accuracy_list_1[i]) / len(accuracy_list_1[i]) for i in range(len(accuracy_list_1))]

    # 利用sns生成的热力图。
    # plt.figure(figsize=(7, 10))
    # sns.heatmap(avg_accuracy_list_1, annot=True, fmt=".2f", cmap="YlGnBu")
    # plt.show()

    colors = ["#f2f5b6", "#d5eda9", "#b3e6a1", "#64ba9d", "#2ca49c", "#008e99", "#007694", "#005e8a", "#00467a", "#012d64", ]

    manual_heatmap(avg_accuracy_list_1, colors, subject_index, mode, col_row_names)
    manual_prediction_error(avg_accuracy_list_1, prediction_list_1, colors, subject_index, mode, col_row_names)

    # plt.show()