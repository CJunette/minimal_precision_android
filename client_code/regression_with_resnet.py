import os
import multiprocessing
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import configs
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
import util_functions


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_tensor_interval(tensor_file_prefix, bool_value, tensor_col_row_name, end_index, start_index=10):
    '''
    由于这次的数据长度不统一，且用户在最开始会容易出现“眨眼”的情况，因此需要做简单的截取。目前的思路是截取第10帧以后的数据。
    :param tensor_file_prefix: 需要被截取文件在col-row之前的前缀。
    :param tensor_col_row_names: col-row的名称构成的list。
    :param end_index: 截取终点，与col-row下的文件夹中的图片数量有关。
    :param start_index: 截取起点，默认为10。
    :return:
    '''
    if bool_value == 0:
        return [], []

    labels = []
    update_tensor_file_names = []
    tensor_file_list = os.listdir(f"{tensor_file_prefix}/{tensor_col_row_name}")
    tensor_file_list.sort(key=lambda x: int(x.replace("capture_", "").replace(".pt", "")))

    label = util_functions.get_row_and_col(tensor_col_row_name)
    # 对label结果进行归一化处理。
    label = (label[0] / (configs.row_num - 1), label[1] / (configs.col_num - 1))

    for repetition_index in range(start_index, end_index):
        update_tensor_file_names.append(f"{tensor_file_prefix}/{tensor_col_row_name}/capture_{repetition_index}.pt")
        labels.append(label)

    return update_tensor_file_names, labels


def select_tensor_and_location(clipped_distant_file_path,
                               clipped_distant_left_eye_file_path, clipped_distant_right_eye_file_path,
                               left_eye_file_path, right_eye_file_path,
                               clipped_distant_location_path, clipped_distant_eye_location_path):
    clipped_distant_file_names = os.listdir(clipped_distant_file_path)
    clipped_distant_file_names.sort(key=util_functions.get_row_and_col)

    left_distant_eye_file_names = os.listdir(clipped_distant_left_eye_file_path)
    left_distant_eye_file_names.sort(key=util_functions.get_row_and_col)

    right_distant_eye_file_names = os.listdir(clipped_distant_right_eye_file_path)
    right_distant_eye_file_names.sort(key=util_functions.get_row_and_col)

    left_eye_file_names = os.listdir(left_eye_file_path)
    left_eye_file_names.sort(key=util_functions.get_row_and_col)

    right_eye_file_names = os.listdir(right_eye_file_path)
    right_eye_file_names.sort(key=util_functions.get_row_and_col)

    clipped_distant_image_paths_1 = []
    distant_left_eye_image_paths_1 = []
    distant_right_eye_image_paths_1 = []
    left_eye_image_paths_1 = []
    right_eye_image_paths_1 = []
    labels_1 = []
    clipped_distant_locations_1 = []
    clipped_distant_eye_locations_1 = []

    for col_row_index in range(len(clipped_distant_file_names)):
        labels_2 = []
        tensor_col_row_name = clipped_distant_file_names[col_row_index]

        clipped_distant_file_num = len(os.listdir(f"{clipped_distant_file_path}/{tensor_col_row_name}"))
        clipped_distant_left_eye_file_num = len(os.listdir(f"{clipped_distant_left_eye_file_path}/{tensor_col_row_name}"))
        clipped_distant_right_eye_file_num = len(os.listdir(f"{clipped_distant_right_eye_file_path}/{tensor_col_row_name}"))
        clipped_left_eye_file_num = len(os.listdir(f"{left_eye_file_path}/{tensor_col_row_name}"))
        clipped_right_eye_file_num = len(os.listdir(f"{right_eye_file_path}/{tensor_col_row_name}"))

        file_num_list = [clipped_distant_file_num,
                         clipped_distant_left_eye_file_num, clipped_distant_right_eye_file_num,
                         clipped_left_eye_file_num, clipped_right_eye_file_num]
        bool_list = [configs.bool_clipped_distant_camera,
                     configs.bool_clipped_distant_left_eye_camera, configs.bool_clipped_distant_right_eye_camera,
                     configs.bool_left_eye_camera, configs.bool_right_eye_camera]
        length_list = []
        for bool_index, bool_value in enumerate(bool_list):
            if bool_value == 1:
                length_list.append(file_num_list[bool_index])

        if min(length_list) < 15:
            print(f"{tensor_col_row_name} has {min(length_list)} files")
        minimal_length = max(min(length_list), configs.tensor_select_end_index) # 这里我原本想让数据有不同长度的，但由于我担心这会引起不同部位训练结果的偏差，因此这里统一设置为14（存放在configs中）。
        # minimal_length = min(min(length_list), configs.tensor_select_end_index)

        # 这里不同输入的labels_2应该都是一样的。
        clipped_distant_image_paths_2, labels_clipped_distant = select_tensor_interval(clipped_distant_file_path, configs.bool_clipped_distant_camera, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)
        clipped_distant_left_eye_image_paths_2, labels_clipped_distant_left = select_tensor_interval(clipped_distant_left_eye_file_path, configs.bool_clipped_distant_left_eye_camera, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)
        clipped_distant_right_eye_image_paths_2, labels_clipped_distant_right = select_tensor_interval(clipped_distant_right_eye_file_path, configs.bool_clipped_distant_right_eye_camera, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)
        left_eye_image_paths_2, labels_left = select_tensor_interval(left_eye_file_path, configs.bool_left_eye_camera, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)
        right_eye_image_paths_2, labels_right = select_tensor_interval(right_eye_file_path, configs.bool_right_eye_camera, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)

        clipped_distant_location_2, label_distant_location = select_tensor_interval(clipped_distant_location_path, 1, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)
        clipped_distant_eye_location_2, label_distant_eye_location = select_tensor_interval(clipped_distant_eye_location_path, 1, tensor_col_row_name, end_index=minimal_length, start_index=configs.tensor_select_start_index)

        labels_list = [labels_clipped_distant, labels_clipped_distant_left, labels_clipped_distant_right,
                       labels_left, labels_right, label_distant_location, label_distant_eye_location]
        for labels in labels_list:
            if labels:
                labels_2 = labels

        clipped_distant_image_paths_1.extend(clipped_distant_image_paths_2)
        distant_left_eye_image_paths_1.extend(clipped_distant_left_eye_image_paths_2)
        distant_right_eye_image_paths_1.extend(clipped_distant_right_eye_image_paths_2)
        left_eye_image_paths_1.extend(left_eye_image_paths_2)
        right_eye_image_paths_1.extend(right_eye_image_paths_2)
        labels_1.extend(labels_2)

        clipped_distant_locations_1.extend(clipped_distant_location_2)
        clipped_distant_eye_locations_1.extend(clipped_distant_eye_location_2)

    return (clipped_distant_image_paths_1,
            distant_left_eye_image_paths_1, distant_right_eye_image_paths_1,
            left_eye_image_paths_1, right_eye_image_paths_1,
            clipped_distant_locations_1, clipped_distant_eye_locations_1,
            labels_1)


def read_clipped_distant_location(df, minimal_length, tensor_col_row_name, columns):
    print(tensor_col_row_name)
    df_selected = df[(df["image_path"].apply(lambda x: x.split("row_")[1].split("-")[0]) == tensor_col_row_name.split("row_")[1].split("-")[0])
                     & (df["image_path"].apply(lambda x: x.split("col_")[1].split("/")[0]) == tensor_col_row_name.split("col_")[1].split("/")[0])
                     & (df["image_path"].apply(lambda x: int(x.split("capture_")[1].split(".jpg")[0])) >= configs.tensor_select_start_index)
                     & (df["image_path"].apply(lambda x: int(x.split("capture_")[1].split(".jpg")[0])) < minimal_length)]

    location_list = []
    for index, row in df_selected.iterrows():
        column_list = []
        for column in columns:
            column_list.append(row[column])
        location_list.append(column_list)
    return location_list


class CustomDataset(Dataset):
    def __init__(self, subject_names, clipped_size=None, train=True):
        self.train = train
        self.clipped_distant_image_paths = []
        self.clipped_left_distant_eye_image_paths = []
        self.clipped_right_distant_eye_image_paths = []
        self.left_eye_image_paths = []
        self.right_eye_image_paths = []
        self.labels = []
        self.clipped_distant_location_paths = []
        self.clipped_distant_eye_location_paths = []

        for subject_index, subject_name in enumerate(subject_names):
            clipped_distant_file_path = f"output_tensor/subject_{subject_name}/clipped_camera_distant"
            left_distant_eye_file_path = f"output_tensor/subject_{subject_name}/clipped_camera_left_eye"
            right_distant_eye_file_path = f"output_tensor/subject_{subject_name}/clipped_camera_right_eye"
            left_eye_file_path = f"output_tensor/subject_{subject_name}/camera_left_eye"
            right_eye_file_path = f"output_tensor/subject_{subject_name}/camera_right_eye"

            clipped_distant_location_path = f"output_tensor/subject_{subject_name}/log_clip"
            clipped_distant_eye_location_path = f"output_tensor/subject_{subject_name}/log_clip_eye"

            # 只取第10帧以后的数据。
            (clipped_distant_image_paths,
             left_distant_eye_image_paths, right_distant_eye_image_paths,
             left_eye_image_paths, right_eye_image_paths,
             clipped_distant_locations, clipped_distant_eye_locations,
             labels) = select_tensor_and_location(clipped_distant_file_path,
                                                  left_distant_eye_file_path, right_distant_eye_file_path,
                                                  left_eye_file_path, right_eye_file_path,
                                                  clipped_distant_location_path, clipped_distant_eye_location_path, )

            self.clipped_distant_image_paths.extend(clipped_distant_image_paths)
            self.clipped_left_distant_eye_image_paths.extend(left_distant_eye_image_paths)
            self.clipped_right_distant_eye_image_paths.extend(right_distant_eye_image_paths)
            self.left_eye_image_paths.extend(left_eye_image_paths)
            self.right_eye_image_paths.extend(right_eye_image_paths)
            self.clipped_distant_location_paths.extend(clipped_distant_locations)
            self.clipped_distant_eye_location_paths.extend(clipped_distant_eye_locations)
            self.labels.extend(labels)

    def __len__(self):
        length_clipped_distant = len(self.clipped_distant_image_paths)
        length_clipped_distant_left_eye = len(self.clipped_left_distant_eye_image_paths)
        length_clipped_distant_right_eye = len(self.clipped_right_distant_eye_image_paths)
        length_left_eye = len(self.left_eye_image_paths)
        length_right_eye = len(self.right_eye_image_paths)
        return max(length_clipped_distant, length_clipped_distant_left_eye, length_clipped_distant_right_eye, length_left_eye, length_right_eye)

    def __getitem__(self, idx):
        # time1 = time.time()
        image_tensor_list = []
        paths_list = [self.clipped_distant_image_paths,
                      self.clipped_left_distant_eye_image_paths,
                      self.clipped_right_distant_eye_image_paths,
                      self.left_eye_image_paths,
                      self.right_eye_image_paths]

        for paths_index, paths in enumerate(paths_list):
            if paths:
                path = paths[idx]
                image = torch.load(path)
                image = image.type(torch.float32)
                image /= 255
                image_tensor_list.append(image)
        # time2 = time.time()

        location_list = []
        if configs.bool_clipped_distant_camera:
            path = self.clipped_distant_location_paths[idx]
            location = torch.load(path)
            location_list.append(location)
        if configs.bool_clipped_distant_left_eye_camera:
            path = self.clipped_distant_eye_location_paths[idx]
            location = torch.load(path)
            location_list.append(location)
        if location_list:
            location_tensor = torch.concat(location_list, dim=0)
        else:
            location_tensor = torch.tensor([], dtype=torch.float32)
        # time3 = time.time()

        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        # time4 = time.time()
        # print(f"image time: {time2 - time1}", f"location time: {time3 - time2}", f"label time: {time4 - time3}")

        return image_tensor_list, location_tensor, label_tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleResNetModel(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_outputs=2):
        super(SimpleResNetModel, self).__init__()
        self.in_planes = 64

        cov_2d_input_num = 3 * sum([configs.bool_clipped_distant_camera,
                                configs.bool_clipped_distant_left_eye_camera,
                                configs.bool_clipped_distant_right_eye_camera,
                                configs.bool_left_eye_camera,
                                configs.bool_right_eye_camera])
        self.conv1 = nn.Conv2d(cov_2d_input_num, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        linear_input_num = 2048
        if configs.bool_clipped_distant_camera:
            linear_input_num += 4
        if configs.bool_clipped_distant_left_eye_camera or configs.bool_clipped_distant_right_eye_camera:
            linear_input_num += 8

        self.linear = nn.Linear(linear_input_num, num_outputs)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, image_tensor_list, location_tensor):
        # time_1 = time.time()
        concatenated_images = torch.concat(image_tensor_list, dim=1)

        out = F.relu(self.bn1(self.conv1(concatenated_images)))
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if configs.bool_clipped_distant_camera or (configs.bool_clipped_distant_left_eye_camera and configs.bool_clipped_distant_right_eye_camera):
            out = torch.cat((out, location_tensor), dim=1)
        out = self.linear(out)
        # time_2 = time.time()
        # print(f"forward time: {time_2 - time_1}")
        return out


class SimpleResNetLightning(SimpleResNetModel, pl.LightningModule):
    def __init__(self):
        super(SimpleResNetLightning, self).__init__()
        self.validation_outputs = []
        self.training_outputs = []

    def training_step(self, batch, batch_idx):
        image_tensor_list, location_tensor, coordinates = batch
        outputs = self(image_tensor_list, location_tensor)
        loss = nn.MSELoss()(outputs, coordinates)
        mse_loss = loss.item()
        self.training_outputs.append(mse_loss)
        device = next(self.parameters()).device
        print(f'Training on {device}')
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensor_list, location_tensor, coordinates = batch
        outputs = self(image_tensor_list, location_tensor)
        loss = nn.MSELoss()(outputs, coordinates)
        mse_loss = loss.item()
        self.validation_outputs.append(mse_loss)
        return {'val_loss': loss}

    def on_train_epoch_end(self):
        avg_mse_loss = np.mean(self.training_outputs)
        self.log('train_mse_loss', avg_mse_loss)

    def on_validation_epoch_end(self):
        avg_mse_loss = np.mean(self.validation_outputs)
        self.log('val_mse_loss', avg_mse_loss)

    def on_train_epoch_start(self):
        self.training_outputs.clear()

    def validation_epoch_start(self):
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=configs.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8),
            'name': 'step_lr_scheduler'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


def prepare_data(training_indices, validation_indices, clipped_size=200):
    train_dataset = CustomDataset(subject_names=training_indices, clipped_size=clipped_size, train=True)
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.num_of_process)

    val_dataset = CustomDataset(subject_names=validation_indices, clipped_size=clipped_size, train=False)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_of_process)

    return train_loader, val_loader


def check_tensor(subject_index_str: str, data_type_str: str):
    file_path = f"output_tensor/subject_{subject_index_str}/{data_type_str}/row_0-col_0/capture_0.pt"
    image = torch.load(file_path)
    # image = image.type(torch.float32)

    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    print("end")


def train_model_in_batches(total_subject_numbers, clipped_size, mode):
    util_functions.set_configs_bool_camera_given_mode(mode)

    set_seed(configs.seed)
    torch.set_num_threads(4)

    for i in range(total_subject_numbers):
        training_indices = [f"{i}.0"]
        validation_indices = [f"{i}.1"]

        train_data_loader, val_data_loader = prepare_data(training_indices, validation_indices, clipped_size)
        num_epochs = configs.num_epochs
        torch.set_float32_matmul_precision('medium')

        # 从头训练。
        model = SimpleResNetLightning()
        checkpoint_callback = ModelCheckpoint(
            monitor='val_mse_loss',
            dirpath=f'models/custom_resnet_regression/subject_{str(i).zfill(3)}_{mode}',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )
        trainer = pl.Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback], accelerator="gpu", devices=configs.gpu_devices)
        trainer.fit(model, train_data_loader, val_data_loader)


def validate_data_of_subject(subject_index: int, model_file_name: str, mode: str = "all_data"):
    '''
    用于对模型进行验证，后续可以用于可视化。TODO 目前还没有进行完整的debug。
    :param subject_index:
    :param model_file_name:
    :param mode:
    :return:
    '''
    # TODO
    util_functions.set_configs_bool_camera_given_mode(mode)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model_path = f"models/custom_resnet_regression/subject_{str(subject_index).zfill(3)}_{mode}/{model_file_name}"
    model = SimpleResNetLightning.load_from_checkpoint(model_path)
    # model.cuda(0)
    model = torch.nn.DataParallel(model, device_ids=configs.gpu_devices).to(device)
    model.eval()

    # prepare validation data
    validation_indices = [f"{subject_index}.1"]
    val_dataset = CustomDataset(subject_names=validation_indices, clipped_size=None, train=False)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_of_process)

    # validate
    loss_list_1 = []
    actual_loss_list_1 = []
    prediction_and_truth_list_1 = []

    for batch_index, batch in enumerate(val_loader):
        image_tensor_list, location_tensor, coordinates = batch
        for image_tensor_index in range(len(image_tensor_list)):
            image_tensor_list[image_tensor_index] = image_tensor_list[image_tensor_index].to(device)
        location_tensor = location_tensor.to(device)
        coordinates = coordinates.to(device)

        outputs = model(image_tensor_list, location_tensor)

        outputs = outputs.cpu().detach().numpy()
        coordinates = coordinates.cpu().detach().numpy()
        loss_list_2 = np.sqrt(np.sum(np.square(outputs - coordinates), axis=1))

        actual_coordinates = coordinates * np.array([configs.row_num - 1, configs.col_num - 1])
        actual_predictions = outputs * np.array([configs.row_num - 1, configs.col_num - 1])
        actual_loss_list_2 = np.sqrt(np.sum(np.square(actual_coordinates - actual_predictions), axis=1))

        prediction_and_truth = [{"prediction": actual_predictions[index], "truth": actual_coordinates[index]} for index in range(len(actual_predictions))]

        loss_list_1.extend(loss_list_2)
        actual_loss_list_1.extend(actual_loss_list_2)
        prediction_and_truth_list_1.extend(prediction_and_truth)

    return loss_list_1, actual_loss_list_1, prediction_and_truth_list_1
