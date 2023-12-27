subject_num = "0.0"
mode = "half_distance"

num_of_process = 16

camera_distant = 2
camera_left_eye = 1
camera_right_eye = 3

col_num = 14
row_num = 29

clipped_image_width = 800
resized_image_width = 500

seed = 0
num_epochs = 100
learning_rate = 0.002
batch_size = 32
gpu_devices = [2, 3]

bool_clipped_distant_camera = 1
bool_clipped_distant_left_eye_camera = 1
bool_clipped_distant_right_eye_camera = 1
bool_left_eye_camera = 1
bool_right_eye_camera = 1
tensor_select_start_index = 4
tensor_select_end_index = 14

# TODO exp中，需要在每次结束录制后确认一下保存的图像数量是否足够，不够的话需要重新处理这个点。
