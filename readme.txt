This is an extension on the master thesis of Hoi: https://github.com/yyhhoi/gait.

The following changes/additions were made:
-
-
-

Original readme: 
The project focuses on training a deep learning embedding layer that can represent the spatio-temporal characteristics of gait sequence

Overall the project involves the following steps:
    1. OpenPose inference from video
    2. Pre-processing part 1
    3. Pre-processing part 2
    4. Training models
    5. Forward pass test data and save as dataframe
    6. Data analysis and visualization

Belows are the descriptions of the project files, as well as the locations of data/labels.

==================== Thesis related: (./) ==========================

./Thesis_analysis.ipynb
    Jupyter notebook that does the analysis and visualization steps, and generate exact same figures as in the Thesis.
    The script only works on the inference results after forward pass the test data (documented in ./scripts/thesis_analysis_workflow.py)

./thesis_results
    Directory that stores figures drawn by ./Thesis_analysis.ipynb, as well as pre-generated videos for interactive plot.

==================== Overall processing/analysis steps (./scripts/) ==========================

./scripts/thesis_analysis_workflow.py
    Script with comments that guide and reproduce all my steps of data processing, model training, forward inference, data analysis and visualization.
    (Must-read)

./scripts/thesis_analysis_script.py
    Script that contains some wrapper functions.

./scripts/openpose_shellscripts/generate_openpose_shellscript_for_FSF.py
    Script to generate bash shell script for running openpose inference commands in a specific docker environment.

==================== Model/training related (./scripts/Spatiotemporal_VAE/) ==========================

./scripts/Spatiotemporal_VAE/Model.py
    Pytorch classes for building up the Variational AutoEncoder (VAE) network architecture for PoseNet, MotionNet and TaskNet.

./scripts/Spatiotemporal_VAE/ConditionalModel.py
    Pytorch classes mainly inherited from Model.py, to add "Conditional labels" to VAE. Also, it includes the addition of
    PhenotypeNet.

./scripts/Spatiotemporal_VAE/Containers.py
    Containers of the model classes that wrap up useful functions for loading/saving model checkpoints, training models, monitoring training progress, data forward pass and evaluation.

./scripts/Spatiotemporal_VAE/model_chkpt
    Directory that stores the checkpoints of the model during training for every epoch.
    It also stores the details of the hyparameters of the model/training.

./scripts/Spatiotemporal_VAE/analysis_scripts/thesis_save_model_outputs.py
    Code for running inference on test data and save the results for analysis and visualization.

==================== Common libraries (./scripts/common/) ==========================

./scripts/common/preprocess.py
    Script for pre-processing part 1. Mainly about cropping bounding box around the right subject, and re-sizing for normalization.
    This part processes videos very slowly.

./scripts/common/feature_extraction.py
    Script for pre-processing part 2. Using the results from part 1, it further normalizes, clips and imputes the data.
    More importantly, it takes the label file and integrate all necessary features into one dataframe.
    This part runs much faster than part 2.

./scripts/common/generator.py
    Generator classes that sample part of the data to feed into the network.

./scripts/common/keypoints_format.py
    All labels/information about the indexes, names and side of the joint keypoints.

./scripts/common/utils.py
    Miscellaneous useful functions for (1) low/high-pass filtering data, (2) tensor-numpy conversion, (3) extended data operation,
    (4) save/write dataframe/json files, (5) sampling files, (6) reading label and (7) conversion between label integers and texts.

./scripts/common/visualisation.py
    Code for visualization, especially for drawing video of motion sequence.

==================== Data/Labels related (./data/) ==========================

/media/dsgz2tb_2/videos_converted
    Directory that stores all RAW videos of gait examination (~160k)
    (This directory had ceased to exist at the time of writing)

./data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle
    Dataframe that contains all the preprocessed data after part-1 and part-2 preprocessing steps

./data/labels/fn_tasks_phenos_validated_rename.pkl
    Labels of a portion of videos. The labels include patient id, task, phenotype, leg length, age, weight...etc

./data/openpose_keypoints
    Keypoints data for each video stored after Openpose Inference.

./data/openpose_visualisation
    Visualization results of Openpose Inference.

./data/preprocessed_keypoints
    Keypoints data stored after part-1 preprocessing.

./data/preprocessed_visualisation
    Visualization of the part-1 preprocessing results.

