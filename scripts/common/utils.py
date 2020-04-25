# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import re
import pickle
import pandas as pd
import torch

def append_lists(data_list, *args):
    try:
        assert len(data_list) == len(args)
    except AssertionError:
        print("data_list should have the same number of elements as *args")
        raise AssertionError
    for val, arg in zip(args):
        arg.append(val)
    return args


def split_arr(arr, stride=10, kernel=128):
    """
    This function split the array "arr" to multiple arrays by sliding over a time window and stride.
    
    Parameters
    ----------
    arr : numpy.darray
        With shape (num_frames, 25, 2)
    stride : int
    kernel : int
    
    Returns
    -------
    split_arr : numpy.darray
        With shape (n_copies, 50, 128)
    """

    num_frames = arr.shape[0]

    if num_frames < (kernel + stride):
        split_arr = np.zeros((1, 50, kernel))
        split_arr[0, 0:25, :] = arr[0:kernel, :, 0].T
        split_arr[0, 25:, :] = arr[0:kernel, :, 1].T
    else:
        n_copies = int((num_frames - kernel) / stride)
        split_arr = np.zeros((n_copies, 50, kernel))
        for i in range(n_copies):
            start = i * stride
            end = kernel + i * stride
            split_arr[i, 0:25, :] = arr[start:end, :, 0].T
            split_arr[i, 25:, :] = arr[start:end, :, 1].T

    return split_arr

class TensorAssigner:
    def __init__(self, size, device):
        self.size = size
        self.helper_tensor, self.finger_print_base = None, None
        self.device = device
        self.clean()

    def assign(self, idx, arr):
        self.helper_tensor[idx, ] = arr

    def get_fingerprint(self):
        return self.finger_print_base * self.helper_tensor

    def clean(self):
        self.helper_tensor = torch.ones(size=self.size, dtype=torch.float).to(self.device)
        self.finger_print_base = torch.ones(size=self.size, dtype=torch.float, requires_grad=True).to(self.device)

class TensorAssignerDouble(TensorAssigner):
    def assign(self, idx1, idx2, arr):
        self.helper_tensor[idx1, idx2, ] = arr

def numpy_bool_index_select(tensor_arr, mask, device, select_dim=0):
    idx = np.where(mask == True)[0]
    idx_tensor = torch.LongTensor(idx).to(device)
    sliced_tensor = tensor_arr.index_select(select_dim, idx_tensor)
    return sliced_tensor


def tensor2numpy(*tensor_arrs):
    output_list = []
    for arr in tensor_arrs:
        output_list.append(arr.cpu().detach().numpy())
    return output_list


def numpy2tensor(device, *numpy_arrs):
    output_list = []
    for arr in numpy_arrs:
        output_list.append(torch.from_numpy(arr).float().to(device))
    return output_list

def slice_by_mask(mask, *arrs):
    new_arrs = []
    for arr in arrs:
        new_arrs.append(arr[mask,])
    return new_arrs

def extract_contagious(binary_train, max_neigh):
    """

    Parameters
    ----------
    binary_train : numpy.darray
        (num_frames, ), with binary values {0, 1}
    max_neigh : int
        Maximum contagious 1's to be grouped

    Returns
    -------
    filtered_binary_arr : numpy.darray
        (num_frames, ). A binary train but with values nerfed to 0's if the 1's have number of neighbours > max_neigh
    group_arr : numpy.darray
        (num_frames, ), each value in the array represents the number of contagious 1's that value neighbours with (including itself.)
    """
    contagious_arr = np.zeros(binary_train.shape)
    group_arr = np.zeros(binary_train.shape)
    filtered_binary_arr = np.zeros(binary_train.shape)
    for i in range(binary_train.shape[0]):

        if binary_train[i] == 1:
            if i == 0:
                contagious_arr[i] = 1
            else:
                contagious_arr[i] = contagious_arr[i - 1] + 1
        else:
            contagious_arr[i] = 0

    focus = contagious_arr[-1]
    for i in reversed(range(binary_train.shape[0])):

        if contagious_arr[i] == 0:
            if i != 0:
                focus = contagious_arr[i - 1]
            else:
                focus = contagious_arr[i]
        else:
            group_arr[i] = focus

    filtered_binary_arr[(group_arr <= max_neigh) & (group_arr > 0)] = 1

    return filtered_binary_arr, (contagious_arr, group_arr)


def pool_points(data, kernel_size):
    """
    Filter out the data space by pooling (select one data point in each kernel window)

    Parameters
    ----------
    data : numpy.darray
        With shape (num_samples, 2) for x-, y-coordinates
    kernel_size : int
        Size of squared kernel. Since skeleton has an aspect ratio of w/h = 1/2, the x-length of the kernel will be halved internally below.

    Returns
    -------
    selected_data_all : numpy.darray
        With shape (num_pooled_samples, 2)
    selected_sampled_index_list : list
        With len = num_samples. Indexes of the selected data with respect to input arg "data".

    """
    max_x, max_y = np.max(data, axis=0)
    min_x, min_y = np.min(data, axis=0)

    kernel_size_x, kernel_size_y = kernel_size / 2, kernel_size

    x_increment_times = int((max_x - min_x) / kernel_size_x) + 1
    y_increment_times = int((max_y - min_y) / kernel_size_y) + 1

    selected_data_list = []
    selected_sampled_index_list = []

    for x_idx in range(x_increment_times):
        for y_idx in range(y_increment_times):
            x_range = (min_x + kernel_size_x * x_idx, min_x + kernel_size_x * (x_idx + 1))
            y_range = (min_y + kernel_size_y * y_idx, min_y + kernel_size_y * (y_idx + 1))

            data_in_range = data[(data[:, 0] > x_range[0]) & (data[:, 0] < x_range[1]) & (data[:, 1] > y_range[0]) & (
                    data[:, 1] < y_range[1])]

            if data_in_range.shape[0] > 0:
                selected_data = np.min(data_in_range, axis=0)
                selected_data_list.append(selected_data)
                selected_sampled_index = np.argmax(np.sum(data == selected_data, axis=1))
                selected_sampled_index_list.append(selected_sampled_index)

    selected_data_all = np.stack(selected_data_list)

    return selected_data_all, selected_sampled_index_list


class RunningAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class MeterAssembly:

    def __init__(self, *args):
        self.meter_dicts = dict()
        self.recorder = dict()
        for arg in args:
            self.meter_dicts[arg] = RunningAverageMeter()
            self.recorder[arg] = []

    def update_meters(self, **kwargs):
        for key in kwargs:
            self.meter_dicts[key].update(kwargs[key])

    def get_meter_avg(self):
        output_dict = dict()
        for key in self.meter_dicts.keys():
            output_dict[key] = self.meter_dicts[key].avg
        return output_dict

    def update_recorders(self):
        for key in self.meter_dicts.keys():
            self.recorder[key].append(self.meter_dicts[key].avg)

    def append_recorders(self, **kwargs):
        for key in kwargs:
            self.recorder[key].append(kwargs[key])

    def get_recorders(self):
        return self.recorder


class OnlineFilter_scalar():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.input_records = np.zeros(kernel_size)
        self.input_times = 0

    def add(self, input_val):
        if self.input_times < self.kernel_size:
            self.input_records[self.input_times] = input_val
            self.input_times += 1
            return np.mean(self.input_records[0:self.input_times])
        else:
            arr_idx = self.input_times % self.kernel_size
            self.input_records[arr_idx] = input_val
            self.input_times += 1
            return np.mean(self.input_records)

    def get_last(self):
        if self.input_times > 0:
            return self.input_records[-1]


class OnlineFilter_np():
    def __init__(self, input_size, kernel_size):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.input_records = self._create_records_arr()
        self.input_times = 0

    def add(self, input_val):
        assert input_val.shape == self.input_size
        if self.input_times < self.kernel_size:
            self.input_records[self.input_times,] = input_val
            self.input_times += 1
            return np.nanmean(self.input_records[0:self.input_times, ], axis=0)
        else:
            arr_idx = self.input_times % self.kernel_size
            self.input_records[arr_idx,] = input_val
            self.input_times += 1
            return np.nanmean(self.input_records, axis=0)

    def get_last(self):
        if self.input_times > 0:
            return self.input_records[-1,]

    def _create_records_arr(self):
        """
        If input_size = (2, 17), and kernel_size = 15, this function will return zero's array of shape (15, 2, 17)
        Hence, input history will always be stored at axis=0
        """
        records_size = [self.kernel_size]
        for i in range(len(self.input_size)):
            records_size.append(self.input_size[i])
        input_records = np.zeros(records_size)
        return input_records


class LabelsReader():
    def __init__(self, labels_path):
        self.labels_path = labels_path
        self.loaded_df = self._read_data_meta_info()
        self.all_filenames = []
        self.output_cols = ["fn_mp4", 'task', "phenotyp_label", "idpatient", "phenotyp_order", "aver_leg"]
        self.vid2task, self.vid2pheno, self.vid2idpatients, self.vid2leg = self._construct_conversion_dict()

    def get_label(self, vid_name_root):

        # Tasks
        try:
            task = task2idx(self.vid2task[vid_name_root])
            task_found = True
        except KeyError:
            task = 0  # shall be masked later
            task_found = False

        # Phenos
        try:
            pheno = pheno2idx(self.vid2pheno[vid_name_root])
            pheno_found = True
        except KeyError:
            pheno = 0  # shall be masked later
            pheno_found = False

        try:
            leg = self.vid2leg[vid_name_root]
            leg_found = True
        except KeyError:
            leg = 0
            leg_found = False

        # idpatients
        idpatient = self.vid2idpatients.get(vid_name_root, None)

        return (task, pheno, idpatient, leg), (task_found, pheno_found, leg_found)


    def get_all_filenames(self):
        return self.all_filenames

    def _construct_conversion_dict(self):
        df_pheno_filtered = self._dataframe_preprocessing()
        self.all_filenames = set(list(df_pheno_filtered["fn_mp4"]))
        df_pheno_filtered["fn_mp4"] = df_pheno_filtered["fn_mp4"].apply(lambda x: os.path.splitext(x)[0])
        vid2task = dict()
        vid2pheno = dict()
        vid2idpatients = dict()
        vid2leg = dict()
        for i in range(df_pheno_filtered.shape[0]):
            vid_name, task, pheno, idpatient, pheno_order, leg = df_pheno_filtered.iloc[i][self.output_cols].copy()
            vid2task[vid_name] = task
            vid2pheno[vid_name] = str(pheno)
            vid2idpatients[vid_name] = idpatient
            vid2leg[vid_name] = leg
        return vid2task, vid2pheno, vid2idpatients, vid2leg

    def _dataframe_preprocessing(self):
        print("Preprocessing dataframe in LabelReader while constructing vid2* convertor dict.")

        # Strip away unnecessary columns
        related_cols = ["fn_mp4", 'task', "phenotyp_label", "idpatient", "phenotyp_order", "leg_length_right", "leg_length_left"]
        df_output = self.loaded_df[related_cols].copy()


        # Calculate the average leg length
        df_output["aver_leg"] = (df_output.leg_length_right + df_output.leg_length_left)/2
        df_output["aver_leg"] = df_output["aver_leg"]/100
        del df_output["leg_length_right"]
        del df_output["leg_length_left"]

        # Strip away phenotype label that is nan
        phenolabel_nan_mask = df_output["phenotyp_label"].astype(str) != 'nan'
        df_output = df_output[phenolabel_nan_mask]

        # Choose phenotype with order of 1 (primary) or nan (seems to have same meaning as 1)
        phenoorder_mask = (df_output["phenotyp_order"] == 1) | (np.isnan(df_output["phenotyp_order"]) == True)
        df_output = df_output[phenoorder_mask]

        # Print and check the shape of dataframe
        print("Before preprocessing, dataframe's shape = \n{}\nAfter preprocessing = \n{}".format(self.loaded_df.shape,
                                                                                                  df_output.shape))

        return df_output

    def _read_data_meta_info(self):
        loaded_df = pd.read_pickle(self.labels_path)
        return loaded_df


def expand1darr(arr, dim, repeat_dim=128):
    """
    Convert 1d numpy array (values parsed to integers) to 2d one-hot vector array.
    Parameters
    ----------
    arr : numpy.darray
        1d-array
    dim : int
        dimension of the label
    """
    m = arr.shape[0]
    output_arr = np.zeros((m, dim, repeat_dim))
    output_arr[np.arange(m), arr, :] = 1
    return output_arr

def expand_1dfloat_arr(arr, repeat_dim=128):
    m = arr.shape[0]
    output_arr = np.ones((m, 1, repeat_dim))
    output_arr[:, :, :] = arr.reshape(m, 1, 1)
    return output_arr

def load_df_pickle(df_path):
    with open(df_path, "rb") as fh:
        try:
            loaded_df = pickle.load(fh, encoding='latin1')
            # loaded_df = pickle.load(fh, encoding='utf-8')
        except TypeError:
            loaded_df = pickle.load(fh)
    return loaded_df


def write_df_pickle(df, write_path):
    with open(write_path, "wb") as fh:
        pickle.dump(df, fh)


def moving_average(arr, kernel_size):
    """
    Unfortunately np.convolve() doesn't provide "boundary replication" padding option. 
    I need to pad the boundary by myself to avoid the boundary artefact of convolution with 0 padding.
    """
    arr_size = arr.shape[0]
    num_to_pad = arr_size - (arr_size - kernel_size + 1)
    if num_to_pad % 2 != 0:
        start_num = np.floor(num_to_pad / 2)
        end_num = np.ceil(num_to_pad / 2)
    else:
        start_num, end_num = num_to_pad / 2, num_to_pad / 2
    start_element, end_element = arr[0], arr[-1]
    front_arr = np.ones(int(start_num)) * start_element
    end_arr = np.ones(int(end_num)) * end_element
    padded_arr = np.concatenate((front_arr, arr, end_arr))
    moving_average = np.convolve(padded_arr, np.ones(kernel_size) / kernel_size, 'valid')
    assert moving_average.shape[0] == arr.shape[0]
    return moving_average


def fullfile(file_path):
    """
    /home/hoi/text.txt = ("text.txt", ("/home/hoi",
                                       "text",
                                       "txt"))
    """

    base_dir, file_name = os.path.split(file_path)
    file_root_name, extension = os.path.splitext(file_name)

    return (file_name, (base_dir, file_root_name, extension))


def dict2json(json_path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def json2dict(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def read_preprocessed_keypoints(data_path):
    return np.load(data_path)['positions_2d'][()]['S1']['Directions 1'].squeeze()


def read_openpose_keypoints(json_path):
    """
    Read openpose's 2D keypoints from .json file. Each json file represents the keypoints of each video frame.

    Args:
        json_path: (str) Path to the .json keypoints file
    Returns:
        keypoints_dict: (dict) Parsed dictionary of the keypoints. 
        num_people: (int) Number of people detected by openpose. It can be equal or greater than 0
        frame_idx: (int) Index of the corresponding video frame, as indicated in the file name of .json
    """
    # Find frame index
    file_base_name = fullfile(json_path)[0]
    frame_idx = int(re.findall('_(\\d{12})_', file_base_name)[0])

    # Find keypoints and number of people
    keypoints_dict = json2dict(json_path)
    num_people = len(keypoints_dict['people'])
    return keypoints_dict, num_people, frame_idx


def read_oenpose_preprocessed_keypoints(np_path):
    return np.load(np_path)['positions_2d']


def read_and_select_openpose_keypoints(json_path):
    """
    Extended from function read_openpose_keypoints(). Read and select the keypoints of the person in the rightest of the frame.

    Args:
        json_path: (str) Path to the .json keypoints file
    Returns:
        keypoints: (ndarray) Numpy array of keypoints with shape (3,25), representing x,y,confidence of the 25 keypoints
    """
    keypoints_dict, num_people, _ = read_openpose_keypoints(json_path)
    if num_people == 0:
        return np.zeros((3, 25))
    elif num_people == 1:
        data_flat = np.asarray(keypoints_dict['people'][0]['pose_keypoints_2d'])
        data = data_flat.reshape(int(len(data_flat) / 3), 3).T
        return data
    elif num_people > 1:
        all_people_keypoints = []
        for i in range(num_people):
            data_flat_each_person = np.asarray(keypoints_dict['people'][i]['pose_keypoints_2d'])
            data_each_person = data_flat_each_person.reshape(int(len(data_flat_each_person) / 3), 3).T  # (3, 25)
            all_people_keypoints.append(data_each_person)
        all_people_keypoints_np = np.asarray(all_people_keypoints)
        person_idx = np.argmax(np.mean(all_people_keypoints_np[:, [0], :], axis=2), axis=0)
        return all_people_keypoints_np[int(person_idx), :, :]


def rename_files(folder, replace_arg):
    from glob import glob
    import os
    all_data = glob(os.path.join(folder, "*"))
    for data in all_data:
        data_root_name = os.path.splitext(os.path.split(data)[1])[0]
        new_name = data_root_name.replace(replace_arg[0], replace_arg[1])
        new_path = os.path.join(folder, new_name)
        os.rename(data, new_path)
        print(data)
        print("renamed to:\n{}\n".format(new_path))


def sample_subset_of_videos(src_dir, sample_num=1000, labels_path="", seed=50, with_labels=True):
    """
    This function sample a number (sample_num, int) of .mp4 videos from source directory (src_dir, str) 
    with a state of randomness (seed, int) and copy them to destination directory (dest_dir, str), 
    and output a log file which records which videos have been copied if write_log == True.
    """
    from glob import glob

    # Grab all videos
    videos_list = glob(os.path.join(src_dir, "*.mp4"))
    filtered_videos_list = []

    # If labels are needed, filter out videos without labels
    if with_labels:
        print("Filtering in those videos with labels... ")
        lreader = LabelsReader(labels_path)
        filenames_with_labels = lreader.get_all_filenames()
        for vid_path_each in videos_list:
            vid_base_name = fullfile(vid_path_each)[0]
            if vid_base_name in filenames_with_labels:
                filtered_videos_list.append(vid_path_each)
    else:
        print("No filtering. Use all videos.")
        filtered_videos_list = videos_list

    # Sample videos
    filtered_videos_list_np = np.array(filtered_videos_list)
    if sample_num > 0:
        print("Sample and copy videos %d from %d" % (sample_num, len(filtered_videos_list_np)))
        np.random.seed(seed)
        selected_videos = np.random.choice(filtered_videos_list_np, sample_num, replace=False)
    else:
        print("Sampling disabled. Use all the videos %d" % len(filtered_videos_list_np))
        selected_videos = filtered_videos_list_np

    return selected_videos


def sample_and_copy_videos(src_dir, dest_dir, sample_num=1000, labels_path="", seed=50, write_log=False,
                           with_labels=True):
    import shutil
    videos_list_np = sample_subset_of_videos(src_dir, sample_num, labels_path, seed, with_labels)
    num_videos = videos_list_np.shape[0]
    # Record the videos copied if needed
    i = 0
    for vid_path in videos_list_np:
        shutil.copy(vid_path, dest_dir)
        if write_log:
            with open("sampled_videos_subset.txt", "a") as fh:
                fh.write("%s" % vid_path)
        i += 1
    print("number of videos copied: %d/%d" % (i, num_videos))


# ['dtcarry' 'dtmath' 'dtspeech' 'ec' 'headneck' 'vmax' 'vmin' 'vself']
# task2idx_dict = {
#     "Vself": 0,
#     "Vmin": 1,
#     "Vmax": 2,
#     "HeadExtended": 3,
#     "DT-speech": 4,
#     "DT-math": 5,
#     "DT-carry": 6,
#     "EC": 7
# }

task2idx_dict = {
    "vself": 0,
    "vmin": 1,
    "vmax": 2,
    "headneck": 3,
    "dtspeech": 4,
    "dtmath": 5,
    "dtcarry": 6,
    "ec": 7
}

idx2task_dict = {v: k for k, v in task2idx_dict.items()}


def idx2task(idx):
    return idx2task_dict[idx]


def task2idx(task):
    return task2idx_dict[task]


# ['ataxia' 'episodic' 'hs' 'hypokinetic' 'normal' 'nph' 'paretic' 'phobic'
#  'ppv' 'psychogenic' 'sensory ataxia' 'spastic' 'suspectnph']

# pheno2idx_dict = {
#     "Anxious":0,
#     "Antalgic":1,
#     "Atactic":2,
#     "Dyskinetic":3,
#     "Functional":4,
#     "Healthy":5,
#     "Hypokinetic-frontal":6,
#     "Hypokinetic":7,
#     "Motor-cognitive":8,
#     "Paretic":9,
#     "Sensory-atactic":10,
#     "Spastic":11,
#     "Spastic-atactic":12
# }

pheno2idx_dict = {
    "aengstliches Gangbild": 0,
    "antalgisches Gangbild": 1,
    "ataktisches Gangbild": 2,
    "dyskinetisches Gangbild": 3,
    "funktionelles Gangbild": 4,
    "gesundes Gangbild": 5,
    "hypokinetisch-frontales Gangbild": 6,
    "hypokinetisches Gangbild": 7,
    "motorisch-kognitives Gangbild": 8,
    "paretisches Gangbild": 9,
    "sensorisch-ataktisches Gangbild": 10,
    "spastisch-ataktisches Gangbild": 11,
    "spastisches Gangbild": 12
}

idx2pheno_dict = {v: k for k, v in pheno2idx_dict.items()}


def idx2pheno(idx):
    return idx2pheno_dict[idx]


def pheno2idx(pheno):
    return pheno2idx_dict[pheno]

# Direction

direction2idx_dict = {
    "Unknown":0,
    "Towards":1,
    "Away":2
}
idx2direction_dict = {v: k for k, v in direction2idx_dict.items()}

def idx2direction(idx):
    return idx2direction_dict[idx]

def direction2idx(idx):
    return direction2idx_dict[idx]

def tick_val_text_tasks():
    vals = list(idx2task_dict.keys())
    texts = [idx2task(i) for i in vals]
    return (vals, texts)

def tick_val_text_phenos():
    vals = list(idx2pheno_dict.keys())
    texts = [idx2pheno(i) for i in vals]
    return (vals, texts)

def tick_val_text_directions():
    vals = list(idx2direction_dict.keys())
    texts = [idx2direction(i) for i in vals]
    return (vals, texts)
