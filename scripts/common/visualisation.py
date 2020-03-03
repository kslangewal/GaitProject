from .keypoints_format import excluded_points, draw_seq_col_indexes
import numpy as np
import matplotlib.pyplot as plt


def build_frame_2by2(*args):
    h, w, _ = args[0].shape
    output_arr = np.zeros((h * 2, w * 2, 3))
    iter_idx, max_iter_idx = 0, len(args) - 1
    for i in range(2):
        for j in range(2):
            if iter_idx <= max_iter_idx:
                output_arr[h * i: h * (i + 1), w * j: w * (j + 1), :] = args[iter_idx]
            iter_idx += 1

    return output_arr


def build_frame_2by3(*args):
    """
    Combine 6 two-dimensional frames with identical shape to a single array (2x3 grid). Starting left to right, top to bottom.

    Parameters
    ----------
    args : numpy.darray
        input frames, each with (h, w, c) shape. Number of arguments <= 6

    Returns
    -------
    output_arr : numpy.darray
        Array combining the <=6 input frames.
    """
    h, w, _ = args[0].shape
    output_arr = np.zeros((h * 2, w * 3, 3))
    iter_idx, max_iter_idx = 0, len(args) - 1
    for i in range(2):
        for j in range(3):
            if iter_idx <= max_iter_idx:
                output_arr[h * i: h * (i + 1), w * j: w * (j + 1), :] = args[iter_idx]
            iter_idx += 1
    return output_arr


def draw_skeleton_new(ax, x, y, linewidth=1):
    for seq_indexes in draw_seq_col_indexes:
        ax.plot(x[seq_indexes[0]], y[seq_indexes[0]], c=seq_indexes[1], linewidth=linewidth)
    return ax


def draw_skeleton_custom(ax, x, y, c, alpha, linewidth=1):
    for seq_indexes in draw_seq_col_indexes:
        ax.plot(x[seq_indexes[0]], y[seq_indexes[0]], c=c, alpha=alpha, linewidth=linewidth)
    return ax


class SkeletonPainter:
    def __init__(self, x, y, texts, sep_x=0.4, y_lim=[-0.6, 0.6]):
        """
        Take in the sequence of x- and y- coordinates of the motions and draw them out.
        The axis 0 of x and y represents the number of skeletons you want to draw. If more than one skeleton, they would
        be alligned horizontally with separation space specified by argument sep_x.

        You need to call the generator self.draw_multiple_skeletons() to yield the numpy array of drawn skeletons for
        each time frame.

        Parameters
        ----------
        x : numpy.darray
            It has shape (m, 25, 128). x-coordinates of the walking sequence of 25 joints
        y : numpy.darray
            It has shape (m, 25, 128). y-coordinates of ^
        texts : tuple or list or iterable
            Labelling texts under the drawn skeletons. It should have length m.
        sep_x : float
            Separation size between the centers of skeletons along x-axis
        y_lim : tuple or list
            Boundaries of y axis of the plot.
        """

        self.x, self.y = x, y
        self.texts = texts
        self.num_skeletons = self.x.shape[0]
        self.sep_x = sep_x
        self.translation_x_vec = np.array([self.sep_x * i for i in range(self.num_skeletons)]).reshape(-1, 1, 1)
        self.text_y = -0.5
        self.x_trans = self.x + self.translation_x_vec
        self.x_trans_flat, self.y_flat, self.concated_excluded_pts = self._concat_excluded_points()
        # Plotting
        self.y_lim = [y_lim[1], y_lim[0]]
        self.x_lim = [np.min(self.translation_x_vec) - 0.2, np.max(self.translation_x_vec) + 0.2]
        self.figsize = (2.2 * self.num_skeletons, 4.8)

    def draw_multiple_skeletons(self):

        for t in range(128):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_ylim(self.y_lim[0], self.y_lim[1])
            ax.set_xlim(self.x_lim[0], self.x_lim[1])

            ax.axis("off")
            for i in range(self.num_skeletons):
                ax = draw_skeleton_new(ax, self.x_trans[i, :, t], self.y[i, :, t])
                ax.scatter(np.delete(self.x_trans_flat[:, t], self.concated_excluded_pts),
                           np.delete(self.y_flat[:, t], self.concated_excluded_pts))
                ax.text(self.translation_x_vec[i, 0, 0], self.text_y, "{}".format(self.texts[i]))

            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            yield data

    def _concat_excluded_points(self):
        excluded_points_np = np.array(excluded_points)
        all_excluded_list = [excluded_points_np + 25 * idx for idx in range(self.num_skeletons)]
        concated_excluded_pts = np.concatenate(all_excluded_list)

        x_trans_flat = self.x_trans.reshape(-1, 128)
        y_flat = self.y.reshape(-1, 128)

        return x_trans_flat, y_flat, concated_excluded_pts


class MotionDrawer:
    def __init__(self, df, motion_cols, name_cols, save_img_path, x_sep, y_sep, interval=[0, 127, 10], figsize=(16, 12),
                 scatter_size=5, dpi=300):
        """

        Parameters
        ----------
        df : pandas.dataframe
        motion_cols : list
            Which model's output you want to draw. Choose at least one to form a list from
            ["ori_motion", "B_recon", "B+C_recon", "B+C+T_recon", "B+C+T+P_recon"]
        name_cols : list
            Label in the picture. Choose at least one to form a list from ["Original", "B", "B+C", "B+C+T", "B+C+T+P"],
            the position of the label must correspond to "motion_cols"
        save_img_path : str
        x_sep : float
            How much the skeletons in the x-axis (time axis) are separated in the drawn figure.
        y_sep : float
            How much the motion sequences between different models (y-axis) are separated in the drawn figure
        interval : list
            For example [0, 128, 10] means drawing from 0th to 128th time frame with 10 increment
        figsize : list or tuple
        scatter_size : float
            Size of the scatter points at the joints
        dpi : int
        """

        # Copy arguments
        self.motion_cols = motion_cols
        self.name_cols = name_cols
        self.save_img_path = save_img_path
        self.interval = interval

        # Data frames/calculation related
        self.num_models = len(self.motion_cols)
        self.df_motions = df[self.motion_cols]
        self.time_indexes = np.arange(*interval)
        self.x_translation = np.arange(self.time_indexes.shape[0]) * x_sep
        self.y_translation = np.arange(self.num_models) * y_sep
        self.margin = 0.2
        self.text_space = 0.5
        self.num_times = self.time_indexes.shape[0]

        # Matplotlib attributes
        self.alpha_offset = 0.3
        self.figsize = figsize
        self.scatter_size = scatter_size
        self.dpi = dpi

        # Reserved
        self.x, self.y = None, None
        self.x_bound, self.y_bound = None, None

    def draw(self, i):
        """

        Parameters
        ----------
        i : int
            Index of the motion sequence: which motion sequence you want to draw?
        """
        # Select and translate skeletons.
        self._convert_selected_data_to_arr(i)
        self._translate_skeletons()
        self._get_plot_boundary()  # Pre-calculate plot window boundarys

        # Plot the motions
        ax = self._plot_motions()

        # Plot's setting
        ax.set_xlim(*self.x_bound)
        ax.set_ylim(*self.y_bound[::-1])
        ax.axis("off")
        plt.tight_layout()

        # Save figure
        if self.save_img_path:
            plt.savefig(self.save_img_path, dpi=self.dpi)
        return ax

    def _convert_selected_data_to_arr(self, i):
        """
        1. Select i index from dataframe.
        2. Stack each (50, 128) motion to (num_models, 50, 128)
        3. Slice (num_models, 50, 128) to (num_models, 50, num_times), "num_times" as defined by the selected interval

        """
        motions_list = []
        for motion_col in self.motion_cols:
            motions_list.append(self.df_motions[motion_col].iloc[i])
        motion_arrs = np.stack(motions_list)
        self.x, self.y = motion_arrs[:, 0:25, self.time_indexes], motion_arrs[:, 25:, self.time_indexes]

    def _translate_skeletons(self):
        self.y = self.y + self.y_translation.reshape(self.y_translation.shape[0], 1, 1)
        self.x = self.x + self.x_translation.reshape(1, 1, self.x_translation.shape[0])

    def _get_plot_boundary(self):
        x_min, x_max = np.min(self.x) - self.text_space, np.max(self.x) + self.margin
        y_min, y_max = np.min(self.y), np.max(self.y)
        self.x_bound, self.y_bound = (x_min, x_max), (y_min, y_max)

    def _plot_motions(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        for m in range(self.num_models):
            for t in range(self.num_times):
                time_portion = (t + 1) / self.num_times * (
                        1 - self.alpha_offset)  # scale the time_portion s.t. time_portion \in [0, 1-1-self.alpha_offset]
                x_to_plot, y_to_plot = self.x[m, :, t], self.y[m, :, t]
                ax = draw_skeleton_custom(ax, x=x_to_plot, y=y_to_plot, c="k", alpha=self.alpha_offset + time_portion)
                ax.scatter(x=np.delete(x_to_plot, excluded_points), y=np.delete(y_to_plot, excluded_points), c="red",
                           s=self.scatter_size, alpha=self.alpha_offset + time_portion)
            ax.text(self.x_bound[0] + 0.1, self.y_translation[m], "{}".format(self.name_cols[m]))
        return ax
