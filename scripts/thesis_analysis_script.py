# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

from Spatiotemporal_VAE.Containers import BaseContainer, ConditionalContainer, PhenoCondContainer
from common.generator import GaitGeneratorFromDFforTemporalVAE
from common.utils import dict2json, json2dict
import os
import pprint


def print_model_info(model_identifier, hyper_params):
    print("%s's hyper-paramters:" % model_identifier)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyper_params)


def load_model_container(model_class, model_identifier, df_path, datagen_batch_size=512, gaitprint_completion=False,
                         train_portion=0.99, seed=0):
    # This function returns an object that wraps over the DL model
    # For each different model identifier, different set of hyperparameters is used
    # To look for the hyper-parameters I used, go to /data/hoi/gait_analysis/scripts/Spatiotemporal_VAE/model_chkpt/
    # Hard-coded stuffs
    seq_dim = 128
    init_lr = 0.001
    lr_milestones = [75]
    lr_decay_gamma = 0.1

    # Hyper-parameters. To be adjusted for each different model.
    hyper_params = {
        "model_name": model_identifier,
        "conditional_label_dim": 3,
        "recon_weight": 1,
        "posenet_latent_dim": 16,
        "posenet_dropout_p": 0,
        "posenet_kld": None,
        "pose_latent_gradient": 0.0001,
        "motionnet_latent_dim": 128,
        "motionnet_dropout_p": 0,
        "motionnet_kld": [0, 10, 0.0001],   # It means, slowly increasing linearly from 0th to 10th epoch from 0 to 0.0001
        "recon_gradient": 0.0001,
        "class_weight": 0.001,
        "latent_recon_loss": 1,
        "fut_weight": 1,
        }
    save_model_path = "Spatiotemporal_VAE/model_chkpt/ckpt_%s.pth" % model_identifier
    save_hyper_params_path = "Spatiotemporal_VAE/model_chkpt/hyperparms_%s.json" % model_identifier

    print('\n\nChecking if path exists')
    if os.path.isfile(save_model_path):
        print("Model checkpoint identified.")
        load_model_path = save_model_path
    else:
        print("Model checkpoint is missing")
        load_model_path = None

    # If hyper-parameters file (identified by filename) already exists, load it. If not, save it.
    if os.path.isfile(save_hyper_params_path):
        print("Existing hyper-params file found")
        hyper_params = json2dict(save_hyper_params_path)
    else:
        print("Hyper-params file saved")
        dict2json(save_hyper_params_path, hyper_params)

    print_model_info(model_identifier, hyper_params)
    if df_path:
        data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=datagen_batch_size, n=seq_dim,
                                                     train_portion=train_portion,
                                                     gait_print=gaitprint_completion, seed=seed)
    else:
        data_gen = None

    # The container object
    model_container = model_class(data_gen=data_gen,
                                  fea_dim=50,
                                  seq_dim=seq_dim,
                                  conditional_label_dim=hyper_params["conditional_label_dim"],
                                  posenet_latent_dim=hyper_params["posenet_latent_dim"],
                                  posenet_dropout_p=hyper_params["posenet_dropout_p"],
                                  posenet_kld=hyper_params["posenet_kld"],
                                  motionnet_latent_dim=hyper_params["motionnet_latent_dim"],
                                  motionnet_hidden_dim=512,
                                  motionnet_dropout_p=hyper_params["motionnet_dropout_p"],
                                  motionnet_kld=hyper_params["motionnet_kld"],
                                  pose_latent_gradient=hyper_params["pose_latent_gradient"],
                                  recon_gradient=hyper_params["recon_gradient"],
                                  classification_weight=hyper_params["class_weight"],
                                  latent_recon_loss=hyper_params["latent_recon_loss"],
                                  fut_weight=hyper_params["fut_weight"],
                                  futnet_hidden_dim=512,
                                  init_lr=init_lr,
                                  lr_milestones=lr_milestones,
                                  lr_decay_gamma=lr_decay_gamma,
                                  save_chkpt_path=save_model_path,
                                  load_chkpt_path=load_model_path)
    return model_container, save_model_path


def run_train_and_vis_on_stvae():
    df_path = "/mnt/data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle"
    training_epoch = 1000
    # Choose the model identifier is one of the four: Thesis_B, Thesis_B+C, Thesis_B+C+T, Thesis_B+C+T+P
    # model_identifier = "Thesis_B"
    # model_identifier = "Thesis_B+C"
    # model_identifier = "Thesis_B+C+T"
    model_identifier = "Thesis_B+C+T+P"

    # =======================================================
    # Based on the model identifier you choose, you will need to the variable identifiers below
    gaitprint_completion = True # True for Thesis B+T+C+P, False for Thesis_B, Thesis_B+C, Thesis_B+C+T
    batch_size = 64  # 64 for Thesis_B+C+T+P, 512 for Thesis_B, Thesis_B+C, Thesis_B+C+T
    # model_class = BaseContainer  # For Thesis_B
    # model_class = ConditionalContainer  # For Thesis_B+C or Thesis_B+C+T
    model_class = PhenoCondContainer  # For Thesis_B+C+T+P
    # =======================================================

    model_container, save_model_path = load_model_container(model_class=model_class,
                                                            model_identifier=model_identifier,
                                                            df_path=df_path,
                                                            datagen_batch_size=batch_size,
                                                            gaitprint_completion=gaitprint_completion,
                                                            train_portion=0.80,
                                                            seed=0)
    # Model checkpoint is automatically saved in every epoch at Spatiotemporal_VAE/model_chkpt/
    model_container.train(training_epoch)


def run_save_model_outputs():
    """
    This function runs forward inference with the four models ("Thesis_B", "Thesis_B+C", "Thesis_B+C+T", "Thesis_B+C+T+P")
    one by one on the test data.

    Returns
    -------
    None
    """
    from Spatiotemporal_VAE.analysis_scripts.thesis_save_model_outputs import OutputSavers
    # Input dataframe
    df_path = "/mnt/data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle"

    # Output dataframes. One for the general results. One for the PhenoNet identificaiton
    df_save_path = "/mnt/thesis_results/data/model_outputs_full_final.pickle"
    df_pheno_save_path = "/mnt/thesis_results/data/model_phenos_outputs_full_final.pickle"

    identifier_set = ["Thesis_B", "Thesis_B+C", "Thesis_B+C+T", "Thesis_B+C+T+P"]
    model_classess = [BaseContainer, ConditionalContainer, ConditionalContainer, PhenoCondContainer]
    model_container_set = []
    data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=512, n=128,
                                                 train_portion=0.8,
                                                 gait_print=False, seed=0)


    for model_identifier, model_class in zip(identifier_set, model_classess):
        model_container_kwargs = {
            "model_class": model_class,
            "model_identifier": model_identifier,
            "df_path": None,
            "datagen_batch_size": 512,
            "gaitprint_completion": False,
            "train_portion": 0.80,
            "seed": 0
        }
        model_container_set.append(model_container_kwargs)

    saver = OutputSavers(data_gen=data_gen,
                         model_container_set=model_container_set,
                         identifier_set=identifier_set,
                         save_df_path=df_save_path,
                         save_pheno_df_path=df_pheno_save_path)

    # Run forward inference here
    saver.forward_batch()
