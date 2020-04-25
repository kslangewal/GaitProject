from glob import glob
from abc import ABC, abstractmethod
from .utils import LabelsReader, fullfile, load_df_pickle
from .keypoints_format import excluded_points_flatten
import random
import os
import numpy as np
import pandas as pd


# Abbreviations:
# SSF = simple shallow features analysis

class DataGenerator(ABC):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.all_data_paths = glob(os.path.join(data_dir, "*"))
        self.num_files = len(self.all_data_paths)

    def iterator(self):
        duration_indices = []
        start = 0
        for stop in range(0, self.num_files, self.batch_size):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop
        random.shuffle(self.all_data_paths)

        for start, stop in duration_indices:
            sampled_data = self._convert_paths_to_data(start, stop)
            yield sampled_data

    @abstractmethod
    def _convert_paths_to_data(self, start, stop):
        return None


class SingleNumpy_DataGenerator(DataGenerator):
    def __init__(self, data_dir, batch_size=1):
        super(SingleNumpy_DataGenerator, self).__init__(data_dir, batch_size)

    def _convert_paths_to_data(self, start, stop):
        batch_data_paths = self.all_data_paths[start:stop]
        for idx, batch_data_path in enumerate(batch_data_paths):
            data_each = np.load(batch_data_path)['positions_2d']

            return (data_each, batch_data_path)


class GaitGeneratorFromDF:

    def __init__(self, df_pickle_path, m=32, n=128, train_portion=0.95, seed=None):
        self.df = load_df_pickle(df_pickle_path)
        self.total_num_rows = self.df.shape[0]
        self.seed = seed
        self.df = self.df.sample(frac=1, random_state=self.seed)
        self.train_portion = train_portion
        self.df_train, self.df_test = self._split_train_test()
        self.num_rows = self.df_train.shape[0]
        self.m, self.n = m, n

        self.label_range = np.max(self.df["tasks"]) - np.min(self.df["tasks"])

    def _split_train_test(self):
        split_index = int(self.total_num_rows * self.train_portion)
        df_train = self.df.iloc[0:split_index, :].reset_index(drop=True)
        df_test = self.df.iloc[split_index:, :].reset_index(drop=True)
        return df_train, df_test

    def iterator(self):
        """
        Randomly sample the indexes from data frame, and yield the sampled batch with the same indexes

        Returns
        -------

        """
        duration_indices = []
        start = 0

        for stop in range(0, self.num_rows, self.m):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop

        if self.seed is not None:
            np.random.seed(self.seed)
        df_shuffled = self.df_train.iloc[np.random.permutation(self.num_rows), :]
        self.seed += 1

        for start, stop in duration_indices:
            info = self._convert_df_to_data(df_shuffled, start, stop)
            yield info

    def _convert_df_to_data(self, df_shuffled, start, stop):

        selected_df = df_shuffled.iloc[start:stop, :].copy()
        output_arr, times = self._loop_for_array_construction(selected_df, self.m)
        output_arr_test, _ = self._loop_for_array_construction(self.df_test, self.m)
        output_arr = output_arr.reshape(self.m, self.n, 25 * 3)
        output_arr_test = output_arr_test.reshape(self.m, self.n, 25 * 3)
        return (output_arr, output_arr_test), times

    def _loop_for_array_construction(self, df, num_samples):
        output_arr = np.zeros((num_samples, self.n, 25, 3))

        for i in range(num_samples):
            # Get features and labels
            fea_vec = df["features"].iloc[i]  # numpy.darray (num_frames, 25, 2)
            label = df["tasks"].iloc[i] / self.label_range  # numpy.int64

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec.shape[0] - self.n)
            fea_vec_sliced = fea_vec[slice_start:slice_start + self.n, :, :]

            # Expand label to match fea_vec_sliced
            label_np = np.ones((self.n, 25)) * label

            # Construct output
            output_arr[i, :, :, 0:2] = fea_vec_sliced
            output_arr[i, :, :, 2] = label_np

        # Construct times
        times = np.arange(self.n) / 25
        return output_arr, times


class GaitGeneratorFromDFforTemporalVAE(GaitGeneratorFromDF):
    """
    Example of usage:

        data_gen = GaitGeneratorFromDFforTemporalVAE(df_pickle_path, m, n, 0.95)

        for (features_train, labels_train), (features_test, labels_test) in data_gen.iterator():
            ...

    where features_train/test has shape (m, num_features=50, n), and labels_train/test has shape (m, )

    """

    def __init__(self, df_pickle_path, m=32, n=128, train_portion=0.95, seed=None, gait_print=False):
        """

        Parameters
        ----------
        df_pickle_path : str
        m : int
            Number of samples drawn from training set for each iteration
        n : int
            Sequence length. Keep it at 128.
        train_portion : float
            Portion of the whole data size to be used as training set. In this class GaitGeneratorFromDFforTemporalVAE,
            it is overridden by setting a particular number and no longer meaningful. See self._split_train_test() method.
        seed : int
            Random seed for data generator.

        """

        # Set number of features
        self.keyps_x_dims, self.keyps_y_dims = 25, 25
        self.fut_dim = 32
        self.total_fea_dims = self.keyps_x_dims + self.keyps_y_dims

        # Call parent's init
        super(GaitGeneratorFromDFforTemporalVAE, self).__init__(df_pickle_path, m, n, train_portion, seed)
        self.batch_shape = (m, self.total_fea_dims, n)
        self.gait_print = gait_print
        self.mt = 4  # number of samples to be drawn for test set

        # Get number of unique patients
        self.num_uni_patients = self._get_num_uni_patients()

        # Construct df filtered out the nan
        self.df_nonan = self._construct_filtered_df()

        self.pheno_stats = []

    def _split_train_test(self):
        labelled_mask = (self.df["task_masks"] == True)
        df_test = self.df[labelled_mask][0:8000].copy()
        train_index = list(self.df[labelled_mask][8000:].index) + list(self.df[labelled_mask == False].index)
        df_train = self.df.loc[train_index].copy()
        return df_train, df_test

    def _convert_df_to_data(self, df_shuffled, start, stop):
        selected_df = df_shuffled.iloc[start:stop, :].copy()

        if self.gait_print:
            selected_df, num_uni_ids_pheno_train = self._complete_gaitprint(selected_df)
            selected_df_test, num_uni_ids_pheno_test = self._complete_gaitprint(self.df_test.sample(n=self.mt))
            #self.pheno_stats = self.pheno_stats + num_uni_ids_pheno_train
        else:
            selected_df_test = self.df_test.sample(n=self.mt)

        # Retrieve train data
        x_train_info, fut_train_info, task_train_info, pheno_train_info, towards_train, leg_train_info, idpatients = self._loop_for_array_construction(
            selected_df,
            selected_df.shape[0])
        x_train, x_train_masks = x_train_info
        fut_train, fut_train_masks, fut_avail_train_mask = fut_train_info
        task_train, task_train_masks = task_train_info
        pheno_train, pheno_train_masks = pheno_train_info
        leg_train, leg_train_masks = leg_train_info

        # Retrieve test data
        x_test_info, fut_test_info, task_test_info, pheno_test_info, towards_test, leg_test_info, idpatients_test = self._loop_for_array_construction(
            selected_df_test,
            selected_df_test.shape[0])

        x_test, x_test_masks = x_test_info
        fut_test, fut_test_masks, fut_avail_test_mask = fut_test_info
        task_test, task_test_masks = task_test_info
        pheno_test, pheno_test_masks = pheno_test_info
        leg_test, leg_test_masks = leg_test_info

        # Combine as output
        train_info = (x_train, x_train_masks, fut_train, fut_train_masks, fut_avail_train_mask, task_train, task_train_masks, pheno_train, pheno_train_masks,
                      towards_train, leg_train, leg_train_masks, idpatients)
        test_info = (x_test, x_test_masks, fut_test, fut_test_masks, fut_avail_test_mask, task_test, task_test_masks, pheno_test, pheno_test_masks,
                     towards_test, leg_test, leg_test_masks, idpatients_test)

        return train_info, test_info

    def _loop_for_array_construction(self, df, num_samples):
        """

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with columns "features" (numpy.darray (num_frames, 25, 2)) and "labels" (numpy.int64)
        num_samples : int
            Size of the sampled data

        Returns
        -------
        input_features : numpy.darray
            It has shape (num_samples, num_features=50, num_time_window), as input array to the network
        labels : numpy.darray
            It has shape (num_samples, ) numpy.int64 [0, 7], as the labels for visualisation

        """
        # fea_vec/fea_mask_vec ~ (num_frames, 25, 2), task ~ int, task_mask ~ bool (True for non-nan, False for nan)
        # pheno ~ int, pheno_mask ~ bool (True for non-nan, False for nan), towards ~ int (0=unknown, 1=left, 2=right)
        # leg ~ float, leg_mask ~ bool (True for non-nan, False for nan), idpatients  int, fut_avail_mask ~ bool (True when future is available)
        select_list = ["features", "feature_masks", "tasks", "task_masks", "phenos", "pheno_masks",
                       "towards_camera", "leg", "leg_masks", "idpatients", "fut_avail_mask"]

        df_np = np.asarray(df[select_list].iloc[0:num_samples])

        fea_vec, fea_mask_vec, task, task_mask, pheno, pheno_mask, towards, leg, leg_mask, idpatients, fut_avail_mask = list(df_np.T)

        task, task_mask = task.astype(np.int), task_mask.astype(np.bool)
        pheno, pheno_mask = pheno.astype(np.int), pheno_mask.astype(np.bool)
        towards, leg, leg_mask = towards.astype(np.int), leg.astype(np.float), leg_mask.astype(np.bool)
        idpatients = idpatients.astype(np.float)
        fut_avail_mask = fut_avail_mask.astype(np.bool)

        features_arr = np.zeros((num_samples, self.total_fea_dims, self.n))
        fea_masks_arr = np.zeros(features_arr.shape)

        fut_features_arr = np.zeros((num_samples, self.total_fea_dims, self.fut_dim))
        fut_fea_masks_arr = np.zeros(fut_features_arr.shape)

        for i in range(num_samples):
            if fut_avail_mask[i]:
                fut_dim = self.fut_dim
            else:
                fut_dim = 0

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec[i,].shape[0] - self.n - fut_dim)
            fea_vec_sliced = fea_vec[i][slice_start:slice_start + self.n, :, :]
            fea_mask_vec_sliced = fea_mask_vec[i][slice_start:slice_start + self.n, :, :]

            # Construct output
            x_end_idx, y_end_idx = self.keyps_x_dims, self.keyps_x_dims + self.keyps_y_dims
            features_arr[i, 0:x_end_idx, :] = fea_vec_sliced[:, :, 0].T  # Store x-coordinates
            features_arr[i, x_end_idx:y_end_idx, :] = fea_vec_sliced[:, :, 1].T  # Store y-coordinates
            fea_masks_arr[i, 0:x_end_idx, :] = fea_mask_vec_sliced[:, :, 0].T  # Store x-coordinates
            fea_masks_arr[i, x_end_idx:y_end_idx, :] = fea_mask_vec_sliced[:, :, 1].T  # Store y-coordinates

            if fut_avail_mask[i]:
                fut_fea_vec_sliced = fea_vec[i][slice_start + self.n:slice_start + self.n + self.fut_dim, :, :]
                fut_fea_mask_vec_sliced = fea_mask_vec[i][slice_start + self.n:slice_start + self.n + self.fut_dim, :, :]

                fut_features_arr[i, 0:x_end_idx, :] = fut_fea_vec_sliced[:, :, 0].T
                fut_features_arr[i, x_end_idx:y_end_idx, :] = fut_fea_vec_sliced[:, :, 1].T
                fut_fea_masks_arr[i, 0:x_end_idx, :] = fut_fea_mask_vec_sliced[:, :, 0].T
                fut_fea_masks_arr[i, x_end_idx:y_end_idx, :] = fut_fea_mask_vec_sliced[:, :, 1].T


        return (features_arr, fea_masks_arr), (fut_features_arr, fut_fea_masks_arr, fut_avail_mask), (task, task_mask), (pheno, pheno_mask), towards, \
               (leg, leg_mask), idpatients

    def _get_num_uni_patients(self):
        idpatients = self.df["idpatients"]
        idpatients_nonan = idpatients[np.isnan(idpatients) == False]
        unique_ids = np.unique(idpatients_nonan)
        self._convert_idpatients_to_index(unique_ids)
        return unique_ids.shape[0]

    def _convert_idpatients_to_index(self, unique_ids):
        # Build conversion dict
        conversion_dict = dict()
        for idx, idpatient in enumerate(unique_ids):
            conversion_dict[idpatient] = idx
        conversion_dict[np.nan] = np.nan
        # Apply conversion to dataframe
        self.df_train["idpatients"] = self.df_train["idpatients"].apply(lambda x: conversion_dict.get(x, np.nan))
        self.df_test["idpatients"] = self.df_test["idpatients"].apply(lambda x: conversion_dict.get(x, np.nan))

    def _complete_gaitprint(self, df):

        id_nan_mask = df["idpatients"].isnull() == False
        current_uni_ids = np.unique(df[id_nan_mask]["idpatients"])

        indexes_to_add = []
        sum_uni_ids_with_pheno = 0
        for uni_id in current_uni_ids:

            df_patient_tasks = self.df_nonan[self.df_nonan["idpatients"] == uni_id]["tasks"]
            grand_id_tasks = np.unique(df_patient_tasks)

            uni_id_tasks = np.unique(df[(df["idpatients"] == uni_id)]["tasks"])

            for grand_id_tasks_each in grand_id_tasks:
                if grand_id_tasks_each not in uni_id_tasks:
                    add_indexes = df_patient_tasks[df_patient_tasks == grand_id_tasks_each].index
                    if add_indexes.shape[0] == 0:
                        continue
                    sampled_add_index = list(np.random.choice(add_indexes, size=1))
                    indexes_to_add += sampled_add_index
            if list(df[(df["idpatients"] == uni_id)]["pheno_masks"])[0] == True :
                sum_uni_ids_with_pheno += 1
        df_to_append = self.df_nonan.loc[indexes_to_add]
        df = pd.concat([df, df_to_append], axis=0)

        return df, sum_uni_ids_with_pheno

    def _construct_filtered_df(self):
        mask = (self.df_train["idpatients"].isnull() == False) & (self.df_train["task_masks"] == True)
        df_nonan = self.df_train[mask]
        return df_nonan
