# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from glob import glob
from .utils import read_oenpose_preprocessed_keypoints, fullfile, LabelsReader, write_df_pickle, load_df_pickle
from .generator import SingleNumpy_DataGenerator
from .keypoints_format import openpose_L_indexes, openpose_R_indexes, openpose_central_indexes
from sklearn.metrics.pairwise import pairwise_distances


class Imputator():
    def __init__(self, data, detect_allnan=False):
        """

        Parameters
        ----------
        data : np.darray
            OpenPose keypoints data of a video sequence. Shape = (num_frames, 25, 3)
        raise_for_allnan : bool
            If True, Raise error for detection of "all nan"" in any keypoint across all video frames.
        """
        self.data = data[:, :, 0:2]  # Exclude confidence terms
        self.detect_allnan = detect_allnan

    def mean_imputation(self):
        """
        Returns:
            data: Numpy array of the same shape as input data with np.nan imputed as the mean
        """
        nan_mask = self._search_for_nan()
        mean_keyps = self._calc_means()
        mean_keyps_expanded = np.ones(self.data.shape) * mean_keyps

        self.data[nan_mask] = mean_keyps_expanded[nan_mask]

        return self.data, nan_mask

    def _search_for_nan(self):
        nan_mask = np.isnan(self.data)  # (num_frames, 25, 2)
        if self.detect_allnan:
            sum_mask = np.sum(nan_mask, axis=0)
            all_nan_deetected = (sum_mask == self.data.shape[0]).any()
        return nan_mask, all_nan_deetected

    def _calc_means(self):
        return np.nanmean(self.data, axis=0, keepdims=True)


class CustomMeanImputator(Imputator):
    def __init__(self, data, mean, detect_allnan=True):
        super(CustomMeanImputator, self).__init__(data, detect_allnan)
        self.mean = mean[:, 0:2].reshape(1, self.data.shape[1], self.data.shape[2])

    def mean_imputation(self):
        """
        Returns:
            data: Numpy array of the same shape as input data with np.nan imputed as the mean
        """
        nan_mask, all_nan_deetected = self._search_for_nan()
        if all_nan_deetected:
            mean_keyps = self._use_existing_means()
        else:
            mean_keyps = self._calc_means()
        mean_keyps_expanded = np.ones(self.data.shape) * mean_keyps
        self.data[nan_mask] = mean_keyps_expanded[nan_mask]
        return self.data, nan_mask

    def _use_existing_means(self):
        return self.mean


class FeatureExtractor():
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.keyps_shape = np.array([25, 3])  # (num_keypoints, keypoint_x_y_confidence)

    def extract(self):
        data_grand_mean = self._incremental_mean_estimation()  # Shape = (25, 3)

        data_gen = SingleNumpy_DataGenerator(self.data_dir, batch_size=1)
        for idx, data_info in enumerate(data_gen.iterator()):
            # Retrive data and print progress
            data, data_path = data_info

            print("Feature Extraction: {}/{} from {}".format(idx, data_gen.num_files, data_path))

            # Imputation
            data_imputed = self._mean_single_imputation(data, data_grand_mean)

            # Clipping (0, 250) and Rescaling (data/250)
            data_imputed = self._clipping_rescaling(data_imputed)

            # Feature extraction work
            extracted_features = self._iterative_workflow(data_imputed)  # Shape = (666,)

            # Saving the data
            assert extracted_features.shape[0] == 666
            save_path = os.path.join(self.save_dir, fullfile(data_path)[1][1])
            np.save(save_path, extracted_features)

    def _incremental_mean_estimation(self):

        data_gen = SingleNumpy_DataGenerator(self.data_dir, batch_size=1)
        num = data_gen.num_files
        data_accumulator = np.zeros([num] + [x for x in self.keyps_shape])

        for idx, data_info in enumerate(data_gen.iterator()):
            print("\r%d/%d Estimating means incrementally from each data file." % (idx, data_gen.num_files), end="",
                  flush=True)
            data, _ = data_info
            data_mean = np.nanmean(data, axis=0)
            data_accumulator[idx] = data_mean
        data_grand_mean = np.nanmean(data_accumulator, axis=0)
        return data_grand_mean

    def _iterative_workflow(self, data):
        """
        Iterative data of each frame of a video, and calcualte the mean and std of relative euclidean distance between keypoints,
        as well as the asymmetry
        Args:
            data: Numpy array with shape (num_frames, 25, 2)
        Returns:
            features: Numpy array with shape (666, )
        """
        features_all_frames = []

        for data_each_frame in data:  # data_each_frame ~ (25, 2)

            # Relative euclidean distance
            pair_dist = pairwise_distances(data_each_frame, metric='euclidean')
            upper_tri_indices = np.triu_indices(n=self.keyps_shape[0], k=1)
            only_relative_dist = pair_dist[upper_tri_indices[0], upper_tri_indices[1]]  # Shape = (300,)

            # Asymmetry
            asymmetry = self._asymmetry_measure(data_each_frame)  # Shape = (33, )

            features_each_frame = np.append(only_relative_dist, asymmetry)  # Shape = (333, )
            features_all_frames.append(features_each_frame)

        features_all_frames_np = np.array(features_all_frames)  # Shape = (num_frames, 333)
        mean, std = np.mean(features_all_frames_np, axis=0), np.std(features_all_frames_np,
                                                                    axis=0)  # Shape = (333,) for both
        features_flattened = np.append(mean, std)  # Shape = (666, )
        return features_flattened

    def _asymmetry_measure(self, data_each_frame):
        """
        Calculate the RATIO of  Euclidean distance of left keypoints to the right keypoints.
        Args:
            data_each_frame: Numpy array with shape (25, 2)
        Returns:
            asymmetry_features: Numpy array with shape (33, )
        """

        def degree_of_asymmetry(L_keyps, R_keyps, anchor):
            """
            Args:
                L_keyps: Numpy array with shape (M, 2)
                R_keyps: Numpy array with shape (M, 2)
                anchor: Numpy array with shape (1, 2), accepting only one anchor vector in each function call.

            """
            L_eu_dist = np.linalg.norm(L_keyps - anchor, axis=1)  # Shape = (M, )
            R_eu_dist = np.linalg.norm(R_keyps - anchor, axis=1)  # Shape = (M, )
            asy_degree = L_eu_dist / (R_eu_dist + 0.00001)  # Shape = (M,)
            return asy_degree

        L_keyps = data_each_frame[openpose_L_indexes, :]  # Shape = (11, 2)
        R_keyps = data_each_frame[openpose_R_indexes, :]  # Shape = (11, 2)
        anchor_points = data_each_frame[openpose_central_indexes, :]  # Shape = (3, 2)
        all_anchors_degree_list = []
        for i in range(anchor_points.shape[0]):
            each_anchor_point = anchor_points[[i], :]  # Shape = (1, 2)
            degrees_each_anchor = degree_of_asymmetry(L_keyps, R_keyps, each_anchor_point)  # Shape = (11,)
            all_anchors_degree_list.append(degrees_each_anchor)

        asymmetry_features = np.array(all_anchors_degree_list).flatten()  # Shape = (11*3, ) = (33, )
        return asymmetry_features

    @staticmethod
    def _mean_single_imputation(data, mean):
        imputor = CustomMeanImputator(data, mean)
        data_imputed, nan_mask = imputor.mean_imputation()
        return data_imputed, nan_mask

    @staticmethod
    def _clipping_rescaling(data, min_val=0, max_val=250, mul_factor=1 / 250):
        data = np.clip(data, min_val, max_val)
        data = data * mul_factor
        return data


class FeatureExtractorForODE(FeatureExtractor):
    """
    The purpose of the class is to generate a dataframe with columns of:
        1. Video names
        2. Feature vectors : np array with (num_frames, 25, 2)
        3. Feature masks : with boolean entries, low confidence keypoints as False, otherwise as True
        4. Tasks : integers [0, 7] for 8 walking tasks. Unlabelled task is set to 0
        5. Task masks : boolean. True = labelled, False = unlabelled
        6. Phenotype : integers [0, 12] for 13 phenotypes. Unlabelled phenotype is set to 0
        7. Phenotype masks : boolean. True = labelled, False = unlabelled
        8. Patient ID
        9. Walkign direction : integers [0, 2]. 0=unknown, 1=towards camera, 2=awayf rom camera
        10. Leg length : float
        11. Leg length masks : boolean. True = labelled, False = unlabelled

    The feature vectors should be a numpy array with shape (num_frames, 25, 2), with below characteristics:
        1. Entries are clipped between [0, 250]
        2. The nan's are imputed by mean of the keypoints across a video (num_frames)
        3. The keypoints' coordinates are scaled to between float [0, 1]

    This class handles input data that were first preprocessed by "OpenposePreprocesser" (Part-1 preprocessing)
    """

    def __init__(self, scr_keyps_dir, labels_path, df_save_path):
        """

        Parameters
        ----------
        scr_keyps_dir : str
            Directory that stored the preprocessed keypoints from Part-1 preprocessing. It should contain a .npz file
            per video

        labels_path : str
            Path that contains the label of z_matrix which can be handled by common.utils.LabelReader class

        df_save_path : str
            Path that you will store your dataframe after this "Part-2 preprocessing"
        """
        self.scr_keyps_dir = scr_keyps_dir
        self.arrs_paths = sorted(glob(os.path.join(self.scr_keyps_dir, "*.npz")))
        self.total_paths_num = len(self.arrs_paths)
        self.df = pd.DataFrame()
        self.lreader = LabelsReader(labels_path)
        self.df_save_path = df_save_path

        # Initialize lists
        self.vid_name_roots_list, self.features_list, self.feature_masks_list = [], [], []
        self.tasks_list, self.task_masks_list = [], []
        self.phenos_list, self.pheno_masks_list = [], []
        self.idpatients_list, self.towards_camera_list = [], []
        self.leg_list, self.leg_mask_list = [], []

        super(FeatureExtractorForODE, self).__init__(scr_keyps_dir, None)
        self.data_grand_mean = self._incremental_mean_estimation()  # Shape = (25, 3)

    def extract(self, filter_window=None):
        """
        Generate the dataframe with the 11 columns as described in class docstring.

        Parameters
        ----------
        filter_window : int
            The minimum size of video frames to be admitted into the dataframe. If None, filtering is disabled.

        Returns
        -------
        None
        """
        for idx, arr_path in enumerate(self.arrs_paths):
            # Print progress
            print("\rSecond preprocessing %d/%d" % (idx, self.total_paths_num), flush=True, end="")

            # Load data
            keyps_arr = np.load(arr_path)["positions_2d"]  # (num_frames, 25, 3)

            # First column: vid_name_root
            vid_name_root = os.path.splitext(os.path.split(arr_path)[1])[0]

            # Second column: features + Forth column: nan_mask
            feature, feature_mask = self._transform_to_features(keyps_arr)
            feature_mask = np.invert(feature_mask)  # False = masked

            # 3rd-5th column: labels
            (task, pheno, idpatient, leg), (task_mask, pheno_mask, leg_mask) = self.lreader.get_label(vid_name_root)

            # Detact walking direction
            towards = self._check_towards(feature,
                                          np.invert(feature_mask))  # For argument here, we want True = masked

            # Append to lists
            self.vid_name_roots_list.append(vid_name_root)
            self.features_list.append(feature)
            self.feature_masks_list.append(feature_mask)  # False = masked
            self.tasks_list.append(task)
            self.task_masks_list.append(task_mask)  # False = masked
            self.phenos_list.append(pheno)
            self.pheno_masks_list.append(pheno_mask)  # False = masked
            self.idpatients_list.append(idpatient)  # None if not found
            self.towards_camera_list.append(towards)  # 0=unknown, 1=towards, 2=away
            self.leg_list.append(leg)
            self.leg_mask_list.append(leg_mask)

        # Create dataframe
        self.df["vid_name_roots"] = self.vid_name_roots_list
        self.df["features"] = self.features_list
        self.df["feature_masks"] = self.feature_masks_list
        self.df["tasks"] = self.tasks_list
        self.df["task_masks"] = self.task_masks_list
        self.df["phenos"] = self.phenos_list
        self.df["pheno_masks"] = self.pheno_masks_list
        self.df["idpatients"] = self.idpatients_list
        self.df["towards_camera"] = self.towards_camera_list
        self.df["leg"] = self.leg_list
        self.df["leg_masks"] = self.leg_mask_list

        # Filter rows with number of frames smaller than "filter_window"
        if (filter_window is not None) and (isinstance(filter_window, int)):
            self._filter(filter_window)

        # # Save dataframe
        write_df_pickle(self.df, self.df_save_path)

    def _filter(self, sequence_window_size=128):
        self.df["num_frames"] = self.df["features"].apply(lambda x: x.shape[0]).copy()
        self.df = self.df[self.df["num_frames"] > sequence_window_size].reset_index(drop=True).copy()
        return None

    def _transform_to_features(self, keyps_arr, boundary=(0, 250)):

        # Clipping (between [0, 250])
        keyps_arr_clipped = np.clip(keyps_arr, boundary[0], boundary[1])

        # Imputation (nan -> mean)
        keyps_imputed, nan_mask = self._mean_single_imputation(keyps_arr_clipped,
                                                               self.data_grand_mean)  # (num_frames, 25, 2)

        # Rescaling (./250)
        keyps_rescaled = keyps_imputed / boundary[1]

        # Translation to hip_centre (index = 8)
        keyps_translated = keyps_rescaled - keyps_rescaled[:, [8], :]

        return keyps_translated, nan_mask

    def _check_towards(self, arr, mask):
        """

        Parameters
        ----------
        arr : numpy.darray
            With shape (num_frames, 25, 2), scaled by ./250 and translated to hip centre
        mask : numpy.darray
            With shape (num_frames, 25, 2), boolean values. True for masked entries.
        Returns
        -------
        towards : bool
            True if the subject is facing towards to camera, False if not
        """

        masked_arr_L = np.ma.masked_array(arr[:, openpose_L_indexes, 0],
                                          mask[:, openpose_L_indexes, 0])  # only x-coordiantes

        masked_arr_R = np.ma.masked_array(arr[:, openpose_R_indexes, 0],
                                          mask[:, openpose_R_indexes, 0])  # only x-coordiantes

        x_sign_L = np.ma.median(np.ma.mean(masked_arr_L, axis=1)) > 0  # Shape -> (num_joints, ) -> (1,)
        x_sign_R = np.ma.median(np.ma.mean(masked_arr_R, axis=1)) > 0  # Shape -> (num_joints, ) -> (1,)

        if (x_sign_L == True) and (x_sign_R == False):
            return 1  # Facing to camera
        elif (x_sign_L == False) and (x_sign_R == True):
            return 2  # Back to camera
        else:
            return 0
