from common.utils import tensor2numpy, write_df_pickle
from thesis_analysis_script import load_model_container
import pandas as pd
import numpy as np
import umap

class OutputSavers:
    """
    This class intends to do inference using exisitng trained models,
    and save inputs + output results for later analysis/visualization in JupyterNotebook

    Belows are the things to be saved in dataframes:

    1. From inputs:
        1.1. Original motion sequence
        1.2. Mask of original motion sequence (True = confident keypoint. False = low confident/non-existing)
        1.3. Task labels, integer between [0,7]
        1.4. Mask of task labels (True = labelled. False = unlabbled)
        1.5. Phenotype labels, integer between [0, 12]
        1.6. Mask of phenotype labels (True = labelled. False = unlabbled)
        1.7. Walking direction, integer between [0, 2]. 0=unknown, 1=towards camera, 2=awayf rom camera

    2. From outputs:
        2.1 Reconstructed motion sequence (for model B, B+C, B+C+T, B+C+T+P)
        2.2 Latent vector (for model B, B+C, B+C+T, B+C+T+P)
        2.3 Latent vector projected onto 2D manifold (for model B, B+C, B+C+T, B+C+T+P)
        2.4 Task prediction (for model B+C+T, B+C+T+P)
        2.5 Phenotype labels (for model B+C+T+P)
        2.6 Phenotype prediction (for model B+C+T+P)

    Except 2.5 and 2.6, all data are written to the dataframe specified by self.save_df_path
    For 2.5 and 2.6, they are written to dataframe specified by self.save_pheno_df_path

    """

    def __init__(self, data_gen, model_container_set, identifier_set, save_df_path, save_pheno_df_path):
        """
        data_gen : object
        model_container_set : list
            List of kwargs dictionaries for models "Thesis_B", "Thesis_B+C", "Thesis_B+C+T", "Thesis_B+C+T+P"
        identifier_set : list
            Expected to be ["Thesis_B", "Thesis_B+C", "Thesis_B+C+T", "Thesis_B+C+T+P"]
        save_df_path : str
            Path for saving the dataframe that stores general data
        save_pheno_df_path : str
            Path for saving the dataframe that stores the results of PhenotypeNet
        """
        self.data_gen = data_gen
        self.data_gen.mt = data_gen.df_test.shape[0]  # s.t. all data are loaded in first generaator loop
        self.identifier_set = [x.replace("Thesis_", "") for x in identifier_set]
        self.model_container_set = model_container_set
        self.df_dict = dict()  # For being loaded into output dataframe (storing general data)
        self.df_pheno_dict = dict()  # For being loaded into output dataframe (storing PhenoType resutls    )
        self.save_df_path = save_df_path
        self.save_pheno_df_path = save_pheno_df_path


    def forward_batch(self):
        # Dummy iteration to dump all test data
        for _, test_data in self.data_gen.iterator():
            pass

        x, nan_masks, tasks_np, tasks_mask_np, phenos_np, phenos_mask_np, towards, _, _, idpatients_np = test_data

        # Store common input data into the output dataframe dictionary
        self.df_dict["ori_motion"] = list(x)
        self.df_dict["ori_motion_mask"] = list(nan_masks)
        self.df_dict["task"] = list(tasks_np)
        self.df_dict["task_mask"] = list(tasks_mask_np)
        self.df_dict["pheno"] = list(phenos_np)
        self.df_dict["pheno_mask"] = list(phenos_mask_np)
        self.df_dict["direction"] = list(towards)

        # Loading each model and doing forward inference in each loop
        for identifier, model_container_kwargs in zip(self.identifier_set, self.model_container_set):
            print("Loading {}".format(identifier))
            model_container, _ = load_model_container(**model_container_kwargs)
            print("forward passing {}".format(identifier))
            data_outputs = model_container.forward_evaluate(test_data)
            if identifier == "B+C+T+P":  # PhenotypeNet has extra columns to store in separate dataframe
                recon, pred_task, _, motion_info, phenos_info, task_latent = data_outputs
                phenos_pred, phenos_labels_np, pheno_latent = phenos_info
            else:
                recon, pred_task, _, motion_info, task_latent = data_outputs
                phenos_pred, phenos_labels_np, pheno_latent = None, None, None

            motion_z, _, _ = motion_info
            data_to_record = (recon, motion_z, pred_task, phenos_pred, phenos_labels_np, pheno_latent, task_latent)

            # Store data into self.df_dict aand self.df_pheno_dict in each loop, for creating an overall dataframe later
            # Umap transformation is also done below
            self._record_data_by_identifier(identifier, data_to_record)

        # Save dataframe
        self._save_dfs()

    def _record_data_by_identifier(self, identifier, data_to_record):

        (recon, motion_z, pred_task, phenos_pred, phenos_labels_np, pheno_latent, task_latent) = data_to_record
        recon, motion_z, pred_task, task_latent = tensor2numpy(recon, motion_z, pred_task, task_latent)

        if identifier == "B+C+T":
            self.df_dict["{}_pred_task".format(identifier)] = list(np.argmax(pred_task, axis=1))
            self.df_dict["{}_task_latent".format(identifier)] = list(task_latent)
            self.df_dict["{}_tl_umap".format(identifier)] = list(self._umap_transform(task_latent))

        elif identifier == "B+C+T+P":
            phenos_pred, pheno_latent = tensor2numpy(phenos_pred, pheno_latent)
            self.df_dict["{}_pred_task".format(identifier)] = list(np.argmax(pred_task, axis=1))
            self.df_dict["{}_task_latent".format(identifier)] = list(task_latent)
            self.df_dict["{}_tl_umap".format(identifier)] = list(self._umap_transform(task_latent))
            self.df_pheno_dict["pheno_pred"] = list(np.argmax(phenos_pred, axis=1))
            self.df_pheno_dict["pheno_labels"] = list(phenos_labels_np)
            self.df_pheno_dict["pheno_latent"] = list(pheno_latent)
            #print('\n\n\n pheno latent: ', list(pheno_latent), '\n\n\n')
            self.df_pheno_dict["pheno_umap"] = list(self._umap_transform(pheno_latent))
        self.df_dict["{}_recon".format(identifier)] = list(recon)
        self.df_dict["{}_z".format(identifier)] = list(motion_z)  # z = latent vector
        self.df_dict["{}_z_umap".format(identifier)] = list(self._umap_transform(motion_z))

    def _save_dfs(self):
        df = pd.DataFrame(self.df_dict)
        df_phenos = pd.DataFrame(self.df_pheno_dict)
        write_df_pickle(df, self.save_df_path)
        write_df_pickle(df_phenos, self.save_pheno_df_path)

    def _umap_transform(self, motion_z):
        umapper = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric="euclidean"
        )
        z_umap = umapper.fit_transform(motion_z)
        return z_umap
