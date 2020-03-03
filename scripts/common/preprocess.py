import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import skvideo.io as skv
from skimage.transform import resize

from .keypoints_format import openpose2detectron_indexes, openpose_L_indexes, openpose_R_indexes
from .utils import fullfile, read_openpose_keypoints, read_and_select_openpose_keypoints, moving_average, \
    OnlineFilter_scalar, OnlineFilter_np, extract_contagious


def reverse_flips(keyps):
    """
    Reverse the flipping artefact
    Parameters
    ----------
    keyps : numpy.darray
        (num_frames, num_keypoitns=25, num_fea=2)

    Returns
    -------

    """
    com_l, com_r = get_com(keyps)  # "com" stands for Centre of Mass

    com_l_sign = np.sign(np.mean(np.sign(com_l[:, 0])))
    if com_l_sign == -1:
        com_l_flipped = (com_l > 0)
        com_r_flipped = (com_r < 0)
    else:
        com_l_flipped = (com_l < 0)
        com_r_flipped = (com_r > 0)

    com_flipped = (com_l_flipped == True) & (com_r_flipped == True)

    com_flipped_filtered, _ = extract_contagious(com_flipped[:, 0], 10)
    com_flipped_filtered = com_flipped_filtered == 1

    keypts_corrected = keyps.copy()
    keypts_flipped = keyps[com_flipped_filtered, :].copy()
    keypts_template = keyps[com_flipped_filtered, :].copy()
    keypts_flipped[:, openpose_R_indexes] = keypts_template[:, openpose_L_indexes]
    keypts_flipped[:, openpose_L_indexes] = keypts_template[:, openpose_R_indexes]
    keypts_corrected[com_flipped_filtered, :] = keypts_flipped
    return keypts_corrected, (com_l, com_r, com_l_flipped, com_r_flipped, com_flipped)


def get_com(keypts_all):
    """

    Parameters
    ----------
    keypts_all : numpy.darray
        (num_frames, num_keypoitns=25, num_fea=2)

    Returns
    -------
    com_l, com_r : numpy.darray
        (num_frames, num_fea=2), centre of mass of left/right shoulder
    """
    keypts_l = keypts_all[:, [5], :] - keypts_all[:, [8], :]
    keypts_r = keypts_all[:, [2], :] - keypts_all[:, [8], :]
    com_l = np.mean(keypts_l, axis=1)
    com_r = np.mean(keypts_r, axis=1)
    return com_l, com_r


def save_data_openpose(save_path, video_size, cut_duration, keypoints_list, records=None):
    """
    Args:
        video_size (tuple): Size of the video (width, height)

        save_path (str): Path for saving the the information

        cut_duration (tuple): (start_index, end_index) between start and end is the duration of the preprocessed video compared to original

        keypoints_list (list): [keypoints_1, keypoints_2, ..., keypoints_k] for video with k frames. 
                                keypoints_k ~ ndarray with shape (17, 3)

        records (list): Optional. For saving records of how to transform keypoints to the new coordinate system, after cropping the video.
                        List of dictionarys [record_1, record_2, ..., record_k] for video with k frames.
                        record_k = {'translation': X, # X = ndarray with shape (2, 1)
                                    'resizing': Y, # Y = float
                                    'bbox': Z } # Z = ndarray with shape (5, )                  
    Returns:
        None
    """

    keypoints_list_np = np.asarray(keypoints_list)  # ndarray with shape (num_frames, 25, 3)
    metadata = {'layout_name': 'body_25', 'num_joints': 25,
                'keypoints_symmetry': [[12, 13, 14, 5, 6, 7], [9, 10, 11, 2, 3, 4]]}
    vid_info = {'video_shape': video_size, 'cut_duration': cut_duration}
    np.savez(save_path, metadata=metadata, positions_2d=keypoints_list_np, vid_info=vid_info, preprocess_info=records)


def find_torso_length_from_keyps(keypoints_xy_input, padding):
    """
    Find the length of the bounding box which is longer than both the 2D dimension spanned by the keypoints
    In body_25 keypoint scheme, index 1, 8, 11, 14 represent neck, hip_centre, right and left ankle

    Parameters
    ==========
    keypoints_xy_input : numpy.darray
        (25, 3)
    padding : int
    """
    keypoints_xy = keypoints_xy_input.copy()
    keypoints_xy_masked = np.ma.array(keypoints_xy, mask=False)
    keypoints_xy_masked.mask[keypoints_xy[:, 2] == 0, :] = True

    ears_centre_xy = np.mean(keypoints_xy_masked[[17, 18], 0:2], axis=0)
    neck_xy = keypoints_xy[1, 0:2]
    hip_centre_xy = keypoints_xy[8, 0:2]

    ankle_centre_xy = np.mean(keypoints_xy[[11, 14], 0:2], axis=0)

    torso_length = np.linalg.norm(ears_centre_xy - neck_xy) + np.linalg.norm(neck_xy - hip_centre_xy) + np.linalg.norm(
        hip_centre_xy - ankle_centre_xy)

    box_half_length = (torso_length / 2) * 1.3 + padding

    return torso_length, box_half_length, [ears_centre_xy, neck_xy, hip_centre_xy, ankle_centre_xy]


class VideoManager():
    def __init__(self, input_video_path, output_video_path):
        self.vreader = skv.FFmpegReader(input_video_path)
        self.vwriter = skv.FFmpegWriter(output_video_path)
        self.vid_name = fullfile(input_video_path)[0]
        self.vid_name_root = fullfile(input_video_path)[1][1]
        self.num_frames, self.vid_h, self.vid_w, self.vid_channels = self.vreader.getShape()

    def __del__(self):
        self.vreader.close()
        self.vwriter.close()


class OpenposePreprocessor(VideoManager):
    def __init__(self, input_video_path, openpose_data_each_video_dir, output_video_path, output_data_path):
        """
        Preprocess each video (raw video file and the corresponding keypoints inferred by OpenPose)

        Parameters
        ----------
        input_video_path : str
        openpose_data_each_video_dir : str
            Each directory has the same name of the video, and holds the keypoints of every video frame as separate file
        output_video_path : str
        output_data_path : str
        """
        self.op_data_each_video_dir = openpose_data_each_video_dir
        self.output_data_path = output_data_path
        self.all_keyps_dicts, self.all_num_people = [], []
        self.keyps_confidence_threshold = -0.2
        self.cut_padding = 0
        self.cut_video_size = (250, 250)
        self.box_length_filter = OnlineFilter_scalar(kernel_size=15)
        self.keypoints_filter = OnlineFilter_np(input_size=(25, 2), kernel_size=3)
        super(OpenposePreprocessor, self).__init__(input_video_path, output_video_path)

    def initialize(self):
        """
        This function does the necessary steps before processing the data frame by frame:
            1. Eliminate the flipping artefact by reversing the wrong coordinates
            2. Check if the number of frames in video match the number of .json files
            3. Find the duration where the whole skeleton is visible, by keypoints' confidence
        Keypoints' confidence is calculated by counting the number of keypoints with <0.01 confidence

        """

        # Check if the number of frames in video match the number of .json files
        self.all_kepys_data_paths = sorted(glob(os.path.join(self.op_data_each_video_dir, "*.json")))
        assert self.num_frames == len(self.all_kepys_data_paths)
        # Initialize lists
        selected_keyps_list = []
        keyps_confidence_list = []
        selected_centres_list = []
        for keyps_path in self.all_kepys_data_paths:
            # Read data and store them into a list, which can be re-used in other sections
            keyps_dict, num_people, _ = read_openpose_keypoints(keyps_path)
            # Select the patient's keypoints, excluding the other human subjects' in the video
            # The below method also calculates the keypoint confidence aggregated from all keypoints' confidence
            selected_centre, selected_keypoints, keyps_confidence = self._return_selected_keypoints(keyps_dict,
                                                                                                    num_people)
            selected_keyps_list.append(selected_keypoints)
            keyps_confidence_list.append(keyps_confidence)
            selected_centres_list.append(selected_centre)

        self.selected_keyps = np.stack(selected_keyps_list)  # (num_frames, 25, 3)
        self.selected_keyps[:, :, 0:2], _ = reverse_flips(self.selected_keyps[:, :, 0:2])
        self.keyps_confidences = np.stack(keyps_confidence_list)  # (num_frames,)
        self.selected_centres = np.stack(selected_centres_list)  # (num_frames, 3)

        # Find the centre coordinates of the skeleton and low-pass filter them.
        kernel_size = 30
        selected_centres_x_smooth = moving_average(self.selected_centres[:, 0], kernel_size)
        selected_centres_y_smooth = moving_average(self.selected_centres[:, 1], kernel_size)
        self.selected_centres_smooth = np.stack(
            [selected_centres_x_smooth, selected_centres_y_smooth]).T  # (num_frames, 2)

        # Extract the continuous duration of the video segment where the whole skeleton is visible
        # based on the keypoint confidence
        self._find_extracted_duration()

    def preprocess(self, write_video=True, plot_keypoints=False):
        """
        Start preprocessing:
        1. Translation of keypoints to the bounding box's cooridnate system
        2. Crop videos to their bounding box of human subject and resize the bounding box to a fixed size
        3. Torso length normalization
        4. Extract only the video segment with whole skeleton visible
        5. Save the videos and processed keypoints

        Parameters
        ----------
        write_video : bool
        plot_keypoints : bool

        Returns
        -------
        None
        """

        all_keypoints = []
        all_records = []
        for frame_idx, vid_frame in enumerate(self.vreader.nextFrame()):
            print("\r{}: {}/{}".format(self.vid_name, frame_idx, self.num_frames), flush=True, end="")

            # Only process those frames within the extracted duration
            if frame_idx < self.start_idx or frame_idx > self.end_idx:
                continue
            else:
                # Find the squared bounding box for cropping. Fixate the crop size and translate the keypoints to
                # bounding box's coordinate system.
                output_frame, keypoints_xy_box_frame, records, box_half_length, central_points = self._find_video_clipping_area(
                    vid_frame, frame_idx)

                # Temporal filtering to the x,y coordinates, but not the confidence
                only_xy = self.keypoints_filter.add(keypoints_xy_box_frame[:, [0, 1]])
                keypoints_xy_box_frame[:, [0, 1]] = only_xy
                all_keypoints.append(keypoints_xy_box_frame)
                all_records.append(records)

                # Plotting (optional)
                if plot_keypoints:
                    new_torso_length, _, _ = find_torso_length_from_keyps(keypoints_xy_box_frame, 0)
                    fig, ax = plt.subplots()
                    ax.imshow(output_frame)
                    ax.scatter(only_xy[openpose_L_indexes, 0], only_xy[openpose_L_indexes, 1], marker="x", c="r")
                    ax.scatter(only_xy[openpose_R_indexes, 0], only_xy[openpose_R_indexes, 1], marker="x", c="b")
                    # central_line = np.vstack(central_points)
                    # ax.plot(central_line[:, 0], central_line[:, 1])
                    fig.suptitle("Torso length:\n%0.2f" % (new_torso_length))
                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.vwriter.writeFrame(data)
                    plt.close()
                elif write_video:
                    self.vwriter.writeFrame(output_frame)

        # Save the keypoints information with the extracted duration
        # for the Part 2 preprocessing (not covered in this class)
        cut_duration = (self.start_idx, self.end_idx)
        save_data_openpose(self.output_data_path, self.cut_video_size, cut_duration, all_keypoints, all_records)

    def _find_extracted_duration(self):
        start_idx_found, end_idx_found = False, False
        self.keyps_confidence_np_filtered = moving_average(self.keyps_confidences, 5)

        for i in range(self.num_frames - 1):
            reverse_i = self.num_frames - 1 - i
            current_confidence_forward = self.keyps_confidence_np_filtered[i]
            current_confidence_backward = self.keyps_confidence_np_filtered[reverse_i]
            if (current_confidence_forward > self.keyps_confidence_threshold) and (start_idx_found == False):
                self.start_idx = i
                start_idx_found = True
            if (current_confidence_backward > self.keyps_confidence_threshold) and (end_idx_found == False):
                self.end_idx = reverse_i
                end_idx_found = True
        return None

    def _return_selected_keypoints(self, keyps_dict, num_people):

        if num_people == 0:
            keyps_selected = np.zeros((25, 3))
            keyps_confidence = -25
            keyps_centre = np.array([0, 0, 0])
            return keyps_centre, keyps_selected, keyps_confidence

        elif num_people == 1:
            keyps_selected = np.asarray(keyps_dict['people'][0]['pose_keypoints_2d']).reshape(25, 3)
            keyps_centre = np.mean(keyps_selected[(keyps_selected[:, 2] > 0.05), :], axis=0)  # (3, )
            keyps_confidence = self.calc_confidence(keyps_selected)
        elif num_people > 1:
            max_x = 0
            max_index = 0
            # Find the person in the most right across all candidates
            for j in range(num_people):
                keyps_np = np.asarray(keyps_dict['people'][j]['pose_keypoints_2d']).reshape(25, 3)
                keyps_np_mask = np.ma.array(keyps_np, mask=False)
                filter_indexes = np.where(keyps_np[:, 2] < 0.005)  # We don't want to include low-confidence keypoints
                keyps_np_mask.mask[filter_indexes[0], :] = True
                mean_x = keyps_np_mask[:, 0].mean()
                if mean_x > max_x:
                    max_x = mean_x
                    max_index = j
            keyps_selected = np.asarray(keyps_dict['people'][max_index]['pose_keypoints_2d']).reshape(25, 3)
            keyps_centre = np.mean(keyps_selected[(keyps_selected[:, 2] > 0.05), :], axis=0)  # (3, )
            keyps_confidence = self.calc_confidence(keyps_selected)

        return keyps_centre, keyps_selected, keyps_confidence

    def _find_video_clipping_area(self, vid_frame, frame_idx):

        output_frame = vid_frame.copy()
        transformation_records = dict()

        # Find the square box length and the torso length. The length of the squared box was determined by torso length.
        keypoints_xy = self.selected_keyps[frame_idx, :, :]  # keypoints_xy (25, 3)
        _, box_length_half, central_points = find_torso_length_from_keyps(keypoints_xy, self.cut_padding)
        if box_length_half == 0:
            box_length_half = self.box_length_filter.get_last()
        else:
            box_length_half = self.box_length_filter.add(box_length_half)  # Temporal filtering

        # Retrieve smoothened bounding box centre and its boundary
        centre_x, centre_y = self.selected_centres_smooth[frame_idx]
        bbox_boundary = np.around(np.array([
            centre_x - box_length_half, centre_y - box_length_half,
            centre_x + box_length_half, centre_y + box_length_half
        ])).astype(int)

        # Ensuring bounding box does not cross the frame boundary
        x_diff, y_diff = 0, 0
        if bbox_boundary[0] < 0:
            x_diff -= bbox_boundary[0]

        if bbox_boundary[1] < 0:
            y_diff -= bbox_boundary[1]

        if bbox_boundary[2] > self.vid_w:
            diff = bbox_boundary[2] - self.vid_w
            x_diff -= diff

        if bbox_boundary[3] > self.vid_h:
            diff = bbox_boundary[3] - self.vid_h
            y_diff -= diff

        # Translation of keypoints
        translation_vec = np.array([bbox_boundary[0], bbox_boundary[1]]).reshape(1, 2)
        keypoints_xy[:, [0, 1]] = keypoints_xy[:, [0, 1]] - translation_vec
        bbox_boundary[0] += x_diff
        bbox_boundary[1] += y_diff
        bbox_boundary[2] += x_diff
        bbox_boundary[3] += y_diff
        transformation_records["translation"] = translation_vec

        # Resize the cropping (bounding box) size to the pre-defined value
        # This also ensures the normalization of torso length, as "bbox_boundary" is determined by torso length
        old_width = bbox_boundary[2] - bbox_boundary[0]
        new_width = self.cut_video_size[0]
        resizing_factor = new_width / old_width
        output_frame = output_frame[bbox_boundary[1]:bbox_boundary[3], bbox_boundary[0]:bbox_boundary[2]]
        output_frame = resize(output_frame, self.cut_video_size, anti_aliasing=True)
        keypoints_xy[:, [0, 1]] = keypoints_xy[:, [0, 1]] * resizing_factor
        transformation_records["resizing"] = resizing_factor
        transformation_records["bbox"] = bbox_boundary
        for i in range(4):
            central_points[i] = (central_points[i] - bbox_boundary[0:2]) * resizing_factor

        # set unconfident data to nan
        keypoints_xy[keypoints_xy[:, 2] < 0.1, :] = np.nan
        keypoints_xy[(keypoints_xy[:, 0] < 0) | (keypoints_xy[:, 0] > self.cut_video_size[0]),
        :] = np.nan
        keypoints_xy[(keypoints_xy[:, 1] < 0) | (keypoints_xy[:, 1] > self.cut_video_size[1]),
        :] = np.nan

        output_frame = np.around(output_frame * 255).astype(int)
        return output_frame, keypoints_xy, transformation_records, box_length_half, central_points

    @staticmethod
    def calc_confidence(keyps_selected):

        base_confidence = 0
        keyps_selected_for_confidence = keyps_selected.copy()
        # Nose and eyes are not included for confidence calculation. The maximum of left/right ears are selected to include.
        keyps_selected_for_confidence[[17, 18], 2] = np.max(keyps_selected[[17, 18], 2])
        # Indexes of nose, right and left_eye are 0,15,16 respectively
        keyps_confidence = base_confidence - np.sum(keyps_selected_for_confidence[1:15, 2] < 0.1) - np.sum(
            keyps_selected_for_confidence[17:, 2] < 0.1)

        return keyps_confidence

    def __del__(self):
        try:
            super(OpenposePreprocessor, self).__del__()
        except AttributeError:
            pass


def openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir,
                                output_data_dir, error_log_path="", plot_keypoints=False,
                                write_video=True):
    """
    This function preprocesses the raw videos and keypoints that were inferred by OpenPose, and output the
    processed keypoints and (optiional) visualization to the designated directories.

    Parameters
    ----------
    src_vid_dir : str
        Directory that you store your raw videos.
    input_data_main_dir : str
        Directory that you store the subfolders of keypoints, inferred from OpenPose.
    output_vid_dir : str
        Directory that you store the output preprocessed visualisation videos.
    output_data_dir : str
        Directory that you store the output preprocessed keypoints.
    error_log_path : str
        Define the path of file that logs the traceback message whenever error is encountered.
    plot_keypoints : bool
        True if you want to plot the keypoints on the output visualisation videos.
        False if you don't do the plotting. The processing will then be faster.
    write_video : bool
        True if you want to store the output preprocessed visualisation videos.
    """
    import traceback

    subfolder_paths = sorted(glob(os.path.join(input_data_main_dir, "*")))
    num_vids = len(subfolder_paths)
    if error_log_path:
        with open(error_log_path, "w") as fh:
            fh.write("\n")

    for idx, subfolder_path_each in enumerate(subfolder_paths):
        try:
            # Monitor progress
            print("\rPreprocessing {}/{}: from {}".format(idx, num_vids, subfolder_path_each))

            # Define paths
            vid_name_root = os.path.split(subfolder_path_each)[1]
            input_video_path = os.path.join(src_vid_dir, vid_name_root + ".mp4")
            output_vid_path = os.path.join(output_vid_dir, vid_name_root + ".mp4")
            output_keypoints_path = os.path.join(output_data_dir, vid_name_root + ".npz")

            # Create output_vid and output_data directory if not exist
            os.makedirs(output_vid_dir, exist_ok=True)
            os.makedirs(output_data_dir, exist_ok=True)

            # Skip if the outputp already exists
            if os.path.isfile(output_keypoints_path):
                print("Skipped: ", vid_name_root)
                continue

            # Start preprocessing
            preprop = OpenposePreprocessor(input_video_path=input_video_path,
                                           openpose_data_each_video_dir=subfolder_path_each,
                                           output_video_path=output_vid_path,
                                           output_data_path=output_keypoints_path)
            preprop.initialize()

            preprop.preprocess(plot_keypoints=plot_keypoints, write_video=write_video)

        except Exception:
            with open(error_log_path, "a") as fh:
                fh.write("\n{}\n".format("=" * 30))
                traceback.print_exc(file=fh)
                print("\nError encountered, logged.")


if __name__ == "__main__":
    pass
