import numpy as np

# left and right is from the owner's perspective, not the observer's

openpose_body25_labels = {
    0: "nose",
    1: "neck",
    2: "r_shoulder",
    3: "r_elbow",
    4: "r_wrist",
    5: "l_shoulder",
    6: "l_elbow",
    7: "l_wrist",
    8: "hip_centre",
    9: "r_hip",
    10:"r_knee",
    11:"r_ankle",
    12:"l_hip",
    13:"l_knee",
    14:"l_ankle",
    15:"r_eye",
    16:"l_eye",
    17:"r_ear",
    18:"l_ear",
    19:"l_bigtoe",
    20:"l_smalltoe",
    21:"l_heel",
    22:"r_bigtoe",
    23:"r_smalltoe",
    24:"r_heel",
}

openpose_body25_indexes = {v: k for k, v in openpose_body25_labels.items()}

openpose_body_connection_scheme = (
    (0, 1),  # nose to neck
    (0, 15),  # nose to r_eye
    (0, 16),  # nose to l_eye
    (15, 17),  # r_eye to r_ear
    (16, 18),  # l_eye to l_ear
    (18, 1),  # l_ear to neck
    (17, 1),  # r_ear to neck
    (1, 5),  # neck to l_shoulder
    (5, 6),  # l_shoulder to l_elbow
    (6, 7),  # l_elbow to l_wrist
    (1, 2),  # neck to r_shoulder
    (2, 3),  # r_shoulder to r_elbow
    (3, 4),  # r_elbow to r_wrist
    (1, 8),  # neck to hip_centre
    (8, 9),  # hip_centre to r_hip
    (9, 10),  # r_hip to r_knee
    (10, 11),  # r_knee to r_ankle
    (11, 24),  # r_ankle to r_heel
    (11, 22),  # r_ankle to r_bigtoe
    (22, 23),  # r_bigtoe to r_smalltoe
    (8, 12),  # hip_centre to l_hip
    (12, 13),  # l_hip to l_knee
    (13, 14),  # l_knee to l_ankle
    (14, 21),  # l_ankle to l_heel
    (14, 19),  # l_ankle to l_bigtoe
    (19, 20)  # l_bigtoe to l_small toe
)

openpose2detectron_indexes = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
openpose_L_indexes = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
openpose_R_indexes = [2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24]
openpose_central_indexes = [0, 1, 8] # From top to bottom. 0=nose, 1=neck, 8=hip_centre

excluded_points = [0, 15, 16]
excluded_points_flatten = [0, 15, 16, 25, 40, 41]

draw_seq1 = [["l_ear", "neck", "l_shoulder", "l_elbow", "l_wrist"], "r"]
draw_seq2 = [["r_ear", "neck", "r_shoulder", "r_elbow", "r_wrist"], "b"]
draw_seq3 = [["neck", "hip_centre", "l_hip", "l_knee", "l_ankle", "l_bigtoe", "l_smalltoe"], "r"]
draw_seq4 = [["neck", "hip_centre", "r_hip", "r_knee", "r_ankle", "r_bigtoe", "r_smalltoe"], "b"]
draw_seq5 = [["l_ankle", "l_heel"], "r"]
draw_seq6 = [["r_ankle", "r_heel"], "b"]
draw_seq_list = [draw_seq1, draw_seq2, draw_seq3, draw_seq4, draw_seq5, draw_seq6]

def convert2indexes_in_list(seq):
    seq_labels, color = seq
    seq_indexes = np.array([openpose_body25_indexes[x] for x in seq_labels])
    return [seq_indexes, color]

draw_seq_col_indexes = list(map(convert2indexes_in_list, draw_seq_list ))


def index2feature_dist(n):
    relative_dists_indexes = np.triu_indices(25, k=1)
    x = relative_dists_indexes[0][n]
    y = relative_dists_indexes[1][n]
    return openpose_body25_labels[x], openpose_body25_labels[y]
