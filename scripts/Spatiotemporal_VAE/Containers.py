import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss

import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
from common.utils import MeterAssembly, numpy2tensor, expand1darr

from .Model import SpatioTemporalVAE
from .ConditionalModel import ConditionalSpatioTemporalVAE, ConditionalPhenotypeSpatioTemporalVAE


class BaseContainer:
    """
    Acronym: B, B+T
    Only PoseNet, MotionNet
    With/Without TaskNet
    Without Conditional, PhenoNet
    """

    def __init__(self,
                 data_gen,
                 fea_dim=50,
                 seq_dim=128,
                 fut_dim=32,
                 conditional_label_dim=0,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=None,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=None,
                 recon_weight=1,
                 fut_weight=1,
                 futnet_hidden_dim=512,
                 pose_latent_gradient=0,
                 recon_gradient=0,
                 classification_weight=0,
                 latent_recon_loss=None,  # None = disabled
                 gpu=0,
                 init_lr=0.001,
                 lr_milestones=[50, 100, 150],
                 lr_decay_gamma=0.1,
                 save_chkpt_path=None,
                 load_chkpt_path=None):

        # Others
        self.epoch = 0
        self.device = torch.device('cuda:{}'.format(gpu))
        self.save_chkpt_path = save_chkpt_path
        self.load_chkpt_path = load_chkpt_path

        # Load parameters
        self.data_gen = data_gen
        self.fea_dim = fea_dim
        self.seq_dim = seq_dim
        self.fut_dim = fut_dim
        self.conditional_label_dim = conditional_label_dim
        self.init_lr = init_lr
        self.lr_milestones = lr_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.posenet_latent_dim = posenet_latent_dim
        self.posenet_dropout_p = posenet_dropout_p
        self.motionnet_latent_dim = motionnet_latent_dim
        self.motionnet_dropout_p = motionnet_dropout_p
        self.motionnet_hidden_dim = motionnet_hidden_dim
        self.recon_weight = recon_weight
        self.fut_weight = fut_weight
        self.pose_latent_gradient = pose_latent_gradient
        self.recon_gradient = recon_gradient
        self.classification_weight = classification_weight
        self.posenet_kld = posenet_kld
        self.motionnet_kld = motionnet_kld
        self.posenet_kld_bool = False if self.posenet_kld is None else True
        self.motionnet_kld_bool = False if self.motionnet_kld is None else True
        self.latent_recon_loss = latent_recon_loss

        self.loss_meter = MeterAssembly(
            "train_total_loss",
            "train_recon",
            "train_fut",
            "train_pose_kld",
            "train_motion_kld",
            "train_recon_grad",
            "train_latent_grad",
            "train_acc",
            "test_total_loss",
            "test_recon",
            "test_fut",
            "test_pose_kld",
            "test_motion_kld",
            "test_recon_grad",
            "test_latent_grad",
            "test_acc"
        )
        self.class_criterion = CrossEntropyLoss(reduction="none")
        # Initialize model, params, optimizer, loss
        if load_chkpt_path is None:
            self.model, self.optimizer, self.lr_scheduler = self._model_initialization()
        else:
            self.model, self.optimizer, self.lr_scheduler = self._load_model()
        #self._save_model()  # Enabled only for renewing newly introduced hyper-parameters

    def forward_decode_only(self, motion_info, towards, num_var_dim, num_datapoints, num_kld):
        motion_z, motion_mu, motion_logvar = motion_info
        towards = torch.from_numpy(expand1darr(towards.astype(np.int64), 3, self.seq_dim)).float().to(self.device)

        kld = torch.mean(-0.5 * (1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp()), dim=0)
        _, sorted_ind = torch.topk(kld, num_kld)

        sorted_ind = [42, 90, 12, 70, 108]

        recon_motion = torch.zeros(num_kld, num_var_dim*num_datapoints, self.fea_dim, motion_z.shape[1])

        for idx, val in enumerate(sorted_ind):
            motion_z_one = motion_z[:,val].view(motion_z.shape[0])
            # Get 90th percentile of motion_z_one
            motion_cpu_copy = motion_z_one.cpu()
            max_kld = np.percentile(motion_cpu_copy.numpy(), 90)
            # Get 10th percentile of motion_z_one
            min_kld = np.percentile(motion_cpu_copy.numpy(), 10)
            motion_z_one.cuda()
            recon_motion[idx,:,:,:] = self.model.decode_only(motion_z, towards, sorted_ind, min_kld, max_kld, num_var_dim, num_datapoints)

        return recon_motion

    def forward_evaluate(self, datagen_tuple):
        self.model.eval()
        with torch.no_grad():
            data_input, data_info = self._convert_input_data(datagen_tuple)
            data_outputs = self.model(*data_input)
        return data_outputs

    def train(self, n_epochs=50):
        try:
            for epoch in range(n_epochs):
                iter_idx = 0
                for train_data, test_data in self.data_gen.iterator():
                    # Clear optimizer's previous gradients
                    self.optimizer.zero_grad()

                    # Retrieve data
                    train_input, train_info = self._convert_input_data(train_data)
                    # test_input, test_info = self._convert_input_data(test_data)

                    # # CV set
                    # self.model.eval()
                    # with torch.no_grad():
                    #     test_outputs = self.model(*test_input)
                    #     loss_test, loss_test_indicators = self.loss_function(test_outputs, test_info)
                    #     self._update_loss_meters(loss_test, loss_test_indicators, train=False)

                    # Train set
                    self.model.train()
                    train_outputs = self.model(*train_input)
                    loss_train, loss_train_indicators = self.loss_function(train_outputs, train_info)
                    self._update_loss_meters(loss_train, loss_train_indicators, train=True)

                    # Print for each iteration
                    self._print_for_each_iter(n_epochs=n_epochs, iter_idx=iter_idx, within_iter=True)

                    # Back-prop
                    loss_train.backward()
                    self.optimizer.step()
                    iter_idx += 1

                # save (overwrite) model file every epoch
                self._print_update_for_each_epoch()
                self._save_model()
                self._plot_loss()

        except KeyboardInterrupt as e:
            torch.cuda.empty_cache()
            self._save_model()
            raise e

    def _update_loss_meters(self, total_loss, indicators, train):

        recon, posekld, motionkld, recongrad, latentgrad, acc, fut_predic = indicators

        if train:
            self.loss_meter.update_meters(
                train_total_loss=total_loss.item(),
                train_recon=recon.item(),
                train_fut=fut_predic.item(),
                train_pose_kld=posekld.item(),
                train_motion_kld=motionkld.item(),
                train_recon_grad=recongrad.item(),
                train_latent_grad=latentgrad.item(),
                train_acc=acc
            )
        else:
            self.loss_meter.update_meters(
                test_total_loss=total_loss.item(),
                test_recon=recon.item(),
                test_fut=fut_predic.item(),
                test_pose_kld=posekld.item(),
                test_motion_kld=motionkld.item(),
                test_recon_grad=recongrad.item(),
                test_latent_grad=latentgrad.item(),
                test_acc=acc
            )

    def _print_for_each_iter(self, n_epochs, iter_idx, within_iter=True):
        # Print Info
        print("\r Epoch %d/%d at iter %d/%d | Recon = %0.8f, %0.8f | KLD = %0.8f, %0.8f | Task Acc = %0.3f, %0.3f | Future=%0.8f, %0.8f" % (
            self.epoch,
            n_epochs,
            iter_idx,
            self.data_gen.num_rows / self.data_gen.m,
            self.loss_meter.get_meter_avg()["train_recon"],
            self.loss_meter.get_meter_avg()["test_recon"],
            self.loss_meter.get_meter_avg()["train_motion_kld"],
            self.loss_meter.get_meter_avg()["test_motion_kld"],
            self.loss_meter.get_meter_avg()["train_acc"],
            self.loss_meter.get_meter_avg()["test_acc"],
            self.loss_meter.get_meter_avg()["train_fut"],
            self.loss_meter.get_meter_avg()["test_fut"]
        ), flush=True, end=""
              )
        if not within_iter:
            print()

    def _print_update_for_each_epoch(self):
        print()
        pprint.pprint(self.loss_meter.get_meter_avg())
        self.loss_meter.update_recorders()
        self.epoch = len(self.loss_meter.get_recorders()["train_total_loss"])
        self.lr_scheduler.step(epoch=self.epoch)

    def _load_model(self):
        checkpoint = torch.load(self.load_chkpt_path)
        print('Loaded ckpt from {}'.format(self.load_chkpt_path))
        # Attributes for model initialization
        self.loss_meter = checkpoint['loss_meter']
        self.epoch = len(self.loss_meter.get_recorders()["train_total_loss"])
        self.fea_dim = checkpoint['fea_dim']
        self.seq_dim = checkpoint['seq_dim']
        self.fut_dim = checkpoint['fut_dim']
        self.conditional_label_dim = checkpoint['conditional_label_dim']
        self.init_lr = checkpoint['init_lr']
        self.lr_milestones = checkpoint['lr_milestones']
        self.lr_decay_gamma = checkpoint['lr_decay_gamma']
        self.posenet_latent_dim = checkpoint['posenet_latent_dim']
        self.posenet_dropout_p = checkpoint['posenet_dropout_p']
        self.motionnet_latent_dim = checkpoint['motionnet_latent_dim']
        self.motionnet_dropout_p = checkpoint['motionnet_dropout_p']
        self.motionnet_hidden_dim = checkpoint['motionnet_hidden_dim']
        self.recon_weight = checkpoint['recon_weight']
        self.fut_weight = checkpoint['fut_weight']
        self.pose_latent_gradient = checkpoint['pose_latent_gradient']
        self.recon_gradient = checkpoint['recon_gradient']
        self.classification_weight = checkpoint['classification_weight']
        self.posenet_kld = checkpoint['posenet_kld']
        self.motionnet_kld = checkpoint['motionnet_kld']
        # self.motionnet_kld = 0.0001
        self.posenet_kld_bool = checkpoint['posenet_kld_bool']
        self.motionnet_kld_bool = checkpoint['motionnet_kld_bool']
        self.latent_recon_loss = checkpoint['latent_recon_loss']

        # Model initialization
        model, optimizer, lr_scheduler = self._model_initialization()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return model, optimizer, lr_scheduler

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'loss_meter': self.loss_meter,
                'fea_dim': self.fea_dim,
                'seq_dim': self.seq_dim,
                'fut_dim': self.fut_dim,
                'conditional_label_dim': self.conditional_label_dim,
                'init_lr': self.init_lr,
                'lr_milestones': self.lr_milestones,
                'lr_decay_gamma': self.lr_decay_gamma,
                'posenet_latent_dim': self.posenet_latent_dim,
                'posenet_dropout_p': self.posenet_dropout_p,
                'motionnet_latent_dim': self.motionnet_latent_dim,
                'motionnet_dropout_p': self.motionnet_dropout_p,
                'motionnet_hidden_dim': self.motionnet_hidden_dim,
                'recon_weight': self.recon_weight,
                'fut_weight': self.fut_weight,
                'pose_latent_gradient': self.pose_latent_gradient,
                'recon_gradient': self.recon_gradient,
                'classification_weight': self.classification_weight,
                'posenet_kld': self.posenet_kld,
                'motionnet_kld': self.motionnet_kld,
                'posenet_kld_bool': self.posenet_kld_bool,
                'motionnet_kld_bool': self.motionnet_kld_bool,
                'latent_recon_loss': self.latent_recon_loss
            }, self.save_chkpt_path)

            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def loss_function(self, model_outputs, inputs_info):
        # Unfolding tuples
        x, nan_masks, fut, fut_mask, fut_avail_mask, tasks, tasks_mask = inputs_info
        recon_motion, pred_labels, recon_fut, pose_info, motion_info, task_latent = model_outputs
        pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar = pose_info
        motion_z, motion_mu, motion_logvar = motion_info

        # Posenet kld
        posenet_kld_multiplier = self._get_interval_multiplier(self.posenet_kld)
        posenet_kld_loss_indicator = -0.5 * torch.mean(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
        posenet_kld_loss = posenet_kld_multiplier * posenet_kld_loss_indicator

        # Motionnet kld
        motionnet_kld_multiplier = self._get_interval_multiplier(self.motionnet_kld)
        motionnet_kld_loss_indicator = -0.5 * torch.mean(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp())
        motionnet_kld_loss = motionnet_kld_multiplier * motionnet_kld_loss_indicator

        # Recon loss
        diff = x - recon_motion
        recon_loss_indicator = torch.mean(nan_masks * (diff ** 2))  # For evaluation
        recon_loss = self.recon_weight * recon_loss_indicator  # For error propagation

        # Future prediction loss
        diff = fut[fut_avail_mask == 1] - recon_fut[fut_avail_mask == 1]
        fut_predic_loss_indicator = torch.mean(fut_mask[fut_avail_mask == 1] * (diff ** 2))
        fut_predic_loss = self.fut_weight * fut_predic_loss_indicator

        # Latent recon loss
        squared_pose_z_seq = ((pose_z_seq - recon_pose_z_seq) ** 2)
        recon_latent_loss_indicator = torch.mean(squared_pose_z_seq)
        recon_latent_loss = 0 if self.latent_recon_loss is None else self.latent_recon_loss * recon_latent_loss_indicator

        # Gradient loss
        nan_mask_negibour_sum = self._calc_gradient_sum(nan_masks)
        gradient_mask = (nan_mask_negibour_sum.int() == 2).float()  # If the adjacent entries are both 1
        recon_grad_loss_indicator = torch.mean(gradient_mask * self._calc_gradient(recon_motion))
        pose_latent_grad_loss_indicator = torch.mean(self._calc_gradient(pose_z_seq))
        recon_grad_loss = self.recon_gradient * recon_grad_loss_indicator
        pose_latent_grad_loss = self.pose_latent_gradient * pose_latent_grad_loss_indicator

        # Classification loss
        class_loss_indicator, acc = self._get_classification_acc(pred_labels, tasks, tasks_mask)
        class_loss = self.classification_weight * class_loss_indicator

        # Combine different losses
        ## KLD has to be set to 0 manually if it is turned off, otherwise it is not numerically stable
        motionnet_kld_loss = 0 if self.motionnet_kld is None else motionnet_kld_loss
        posenet_kld_loss = 0 if self.posenet_kld is None else posenet_kld_loss
        loss = recon_loss + posenet_kld_loss + motionnet_kld_loss + recon_grad_loss + pose_latent_grad_loss + \
               recon_latent_loss + class_loss + fut_predic_loss_indicator

        return loss, (
            recon_loss_indicator, posenet_kld_loss_indicator, motionnet_kld_loss_indicator, recon_grad_loss_indicator,
            pose_latent_grad_loss_indicator, acc, fut_predic_loss_indicator)

    def _model_initialization(self):
        model = SpatioTemporalVAE(
            fea_dim=self.fea_dim,
            seq_dim=self.seq_dim,
            fut_dim=self.fut_dim,
            posenet_latent_dim=self.posenet_latent_dim,
            posenet_dropout_p=self.posenet_dropout_p,
            posenet_kld=self.posenet_kld_bool,
            motionnet_latent_dim=self.motionnet_latent_dim,
            motionnet_hidden_dim=self.motionnet_hidden_dim,
            motionnet_dropout_p=self.motionnet_dropout_p,
            motionnet_kld=self.motionnet_kld_bool
        ).to(self.device)

        params = model.parameters()
        optimizer = optim.Adam(params, lr=self.init_lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_decay_gamma)
        return model, optimizer, lr_scheduler

    def _plot_loss(self):
        '''
            "train_recon",
            "train_pose_kld",
            "train_motion_kld",
            "train_recon_grad",
            "train_latent_grad",
            "train_acc",
            "train_fut",
        '''

        def plot_ax_train_test(ax, x_length, windows, recorders, key_suffix, train_ylabel, test_ylabel):
            # ax_tw = ax.twinx()
            ax.plot(x_length, recorders["train_" + key_suffix][windows:], c="b")
            # ax_tw.plot(x_length, recorders["test_" + key_suffix][windows:], c="r")
            ax.set_ylabel(train_ylabel)
            # ax_tw.set_ylabel(test_ylabel)

        def sliding_plot(epoch_windows, axes, recorders):
            windows = self.epoch - epoch_windows
            x_length = np.linspace(windows, self.epoch - 1, epoch_windows)

            plot_ax_train_test(axes[0, 0], x_length, windows, recorders, "recon", "Train Recon MSE", "")
            plot_ax_train_test(axes[1, 0], x_length, windows, recorders, "pose_kld", "Train pose_kld", "")
            plot_ax_train_test(axes[2, 0], x_length, windows, recorders, "motion_kld", "Train motion_kld", "")
            plot_ax_train_test(axes[0, 1], x_length, windows, recorders, "recon_grad", "", "Test recon_grad")
            plot_ax_train_test(axes[1, 1], x_length, windows, recorders, "latent_grad", "", "Test latent_grad")
            plot_ax_train_test(axes[2, 1], x_length, windows, recorders, "acc", "", "Test acc")
            plot_ax_train_test(axes[0, 2], x_length, windows, recorders, "fut", "Train Future MSE", "")

        epoch_windows = 100
        recorders = self.loss_meter.get_recorders()
        fig, ax = plt.subplots(3, 3, figsize=(16, 8))

        # Restrict to show only recent epochs
        if self.epoch > epoch_windows:
            sliding_plot(epoch_windows, ax, recorders)
        else:
            sliding_plot(self.epoch, ax, recorders)

        fig.suptitle(os.path.splitext(os.path.split(self.save_chkpt_path)[1])[0])
        plt.savefig(self.save_chkpt_path + ".png", dpi=300)

    def _convert_input_data(self, data_tuple):
        # Unfolding
        x, nan_masks, fut_np, fut_mask_np, fut_avail_mask_np, tasks, tasks_mask, _, _, _, _, _, _ = data_tuple

        # Convert numpy to torch.tensor
        x, fut = numpy2tensor(self.device, x, fut_np)
        tasks = torch.from_numpy(tasks).long().to(self.device)
        tasks_mask = torch.from_numpy(tasks_mask * 1 + 1e-5).float().to(self.device)
        nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
        fut_mask = torch.from_numpy(fut_mask_np * 1 + 1e-5).float().to(self.device)
        fut_avail_mask = torch.from_numpy(fut_avail_mask_np.astype(int)).to(self.device)

        # Construct tuple
        input_data = (x, fut_np, fut_mask_np)
        input_info = (x, nan_masks, fut, fut_mask, fut_avail_mask, tasks, tasks_mask)
        return input_data, input_info

    def _get_interval_multiplier(self, quantity_arg):
        """

        Parameters
        ----------
        quantity_arg : int or float or list
            Multiplier. If list, e.g. [50, 100, 0.1], then the function returns 0 before self.epoch < 50,
            the returned value linearly increases from 0 to 0.1 between 50th and 100th epoch, and remains as 0.1 after self.epoch > 100
        Returns
        -------
        quantity_multiplier : int or float

        """

        if quantity_arg is None:
            quantity_multiplier = 0
        elif isinstance(quantity_arg, list):
            start, end, const = quantity_arg[0], quantity_arg[1], quantity_arg[2]
            if self.epoch < start:
                quantity_multiplier = 0
            elif (self.epoch >= start) and (self.epoch < end):
                quantity_multiplier = const * ((self.epoch - start) / (end - start))
            elif self.epoch >= end:
                quantity_multiplier = const
        elif isinstance(quantity_arg, int) or isinstance(quantity_arg, float):
            quantity_multiplier = quantity_arg
        return quantity_multiplier

    @staticmethod
    def _calc_gradient(x):
        grad = torch.abs(x[:, :, 0:127] - x[:, :, 1:])
        return grad

    @staticmethod
    def _calc_gradient_sum(x):
        grad = x[:, :, 0:127] + x[:, :, 1:]
        return grad

    def _get_classification_acc(self, pred_labels, labels, label_masks):
        class_loss_indicator_vec = self.class_criterion(pred_labels, labels)
        # class_loss_indicator = torch.mean(class_loss_indicator_vec)
        class_loss_indicator = torch.mean(label_masks * class_loss_indicator_vec)
        if class_loss_indicator is None:
            import pdb
            pdb.set_trace()
        pred_labels_np, labels_np = pred_labels.cpu().detach().numpy(), labels.cpu().detach().numpy()
        label_masks_np = label_masks.cpu().detach().numpy()
        acc = np.mean(np.argmax(pred_labels_np[label_masks_np > 0.5,], axis=1) == labels_np[label_masks_np > 0.5]) * 100
        return class_loss_indicator, acc

    def save_model_losses_data(self, project_dir, model_identifier):
        import pandas as pd
        loss_data = self.loss_meter.get_recorders()
        df_losses = pd.DataFrame(loss_data)
        df_losses.to_csv(os.path.join(project_dir, "vis", model_identifier, "loss_{}.csv".format(model_identifier)))


class ConditionalContainer(BaseContainer):
    """
    Acronym: B+T+C
    Only PoseNet, MotionNet, Conditional
    Without PhenoNet
    """
    def _model_initialization(self):
        model = ConditionalSpatioTemporalVAE(
            fea_dim=self.fea_dim,
            seq_dim=self.seq_dim,
            fut_dim=self.fut_dim,
            posenet_latent_dim=self.posenet_latent_dim,
            posenet_dropout_p=self.posenet_dropout_p,
            posenet_kld=self.posenet_kld_bool,
            motionnet_latent_dim=self.motionnet_latent_dim,
            motionnet_hidden_dim=self.motionnet_hidden_dim,
            motionnet_dropout_p=self.motionnet_dropout_p,
            motionnet_kld=self.motionnet_kld_bool,
            conditional_label_dim=self.conditional_label_dim
        ).to(self.device)
        params = model.parameters()
        optimizer = optim.Adam(params, lr=self.init_lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_decay_gamma)
        return model, optimizer, lr_scheduler

    def _convert_input_data(self, data_tuple):
        # Unfolding
        x, nan_masks, fut_np, fut_mask_np, fut_avail_mask_np, tasks, tasks_mask, _, _, towards, _, _, _ = data_tuple

        # Convert numpy to torch.tensor
        tasks = torch.from_numpy(tasks).long().to(self.device)
        tasks_mask = torch.from_numpy(tasks_mask * 1 + 1e-5).float().to(self.device)
        nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
        fut_mask = torch.from_numpy(fut_mask_np * 1 + 1e-5).float().to(self.device)
        fut_avail_mask = torch.from_numpy(fut_avail_mask_np.astype(int)).to(self.device)
        x, fut, towards = numpy2tensor(self.device,
                               x,
                               fut_np,
                               expand1darr(towards.astype(np.int64), 3, self.seq_dim)
                               )

        # Construct tuple
        input_data = (x, towards, fut_np, fut_mask_np)
        input_info = (x, nan_masks, fut, fut_mask, fut_avail_mask, tasks, tasks_mask)
        return input_data, input_info

class PhenoCondContainer(BaseContainer):
    def __init__(self,
                 data_gen,
                 fea_dim=50,
                 seq_dim=128,
                 fut_dim=32,
                 conditional_label_dim=0,
                 num_phenos=13,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=None,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=None,
                 recon_weight=1,
                 fut_weight=1,
                 futnet_hidden_dim=512,
                 pose_latent_gradient=0,
                 recon_gradient=0,
                 classification_weight=0,
                 latent_recon_loss=None,  # None = disabled
                 gpu=0,
                 init_lr=0.001,
                 lr_milestones=[50, 100, 150],
                 lr_decay_gamma=0.1,
                 save_chkpt_path=None,
                 load_chkpt_path=None):
        self.num_phenos = num_phenos
        super(PhenoCondContainer, self).__init__(
            data_gen=data_gen,
            fea_dim=fea_dim,
            seq_dim=seq_dim,
            fut_dim=fut_dim,
            conditional_label_dim=conditional_label_dim,
            posenet_latent_dim=posenet_latent_dim,
            posenet_dropout_p=posenet_dropout_p,
            posenet_kld=posenet_kld,
            motionnet_latent_dim=motionnet_latent_dim,
            motionnet_hidden_dim=motionnet_hidden_dim,
            motionnet_dropout_p=motionnet_dropout_p,
            motionnet_kld=motionnet_kld,
            recon_weight=recon_weight,
            fut_weight=fut_weight,
            futnet_hidden_dim=futnet_hidden_dim,
            pose_latent_gradient=pose_latent_gradient,
            recon_gradient=recon_gradient,
            classification_weight=classification_weight,
            latent_recon_loss=latent_recon_loss,  # None =latent_recon_loss=None,  # None d
            gpu=gpu,
            init_lr=init_lr,
            lr_milestones=lr_milestones,
            lr_decay_gamma=lr_decay_gamma,
            save_chkpt_path=save_chkpt_path,
            load_chkpt_path=load_chkpt_path
        )
        self.loss_meter = MeterAssembly(
            "train_total_loss",
            "train_recon",
            "train_fut",
            "train_pose_kld",
            "train_motion_kld",
            "train_recon_grad",
            "train_latent_grad",
            "train_acc",
            "train_phenos_loss",
            "train_phenos_acc",
            "test_total_loss",
            "test_recon",
            "test_fut",
            "test_pose_kld",
            "test_motion_kld",
            "test_recon_grad",
            "test_latent_grad",
            "test_acc",
            "test_phenos_loss",
            "test_phenos_acc"
        )

    def _model_initialization(self):
        model = ConditionalPhenotypeSpatioTemporalVAE(
            fea_dim=self.fea_dim,
            seq_dim=self.seq_dim,
            fut_dim=self.fut_dim,
            posenet_latent_dim=self.posenet_latent_dim,
            posenet_dropout_p=self.posenet_dropout_p,
            posenet_kld=self.posenet_kld_bool,
            motionnet_latent_dim=self.motionnet_latent_dim,
            motionnet_hidden_dim=self.motionnet_hidden_dim,
            motionnet_dropout_p=self.motionnet_dropout_p,
            motionnet_kld=self.motionnet_kld_bool,
            conditional_label_dim=self.conditional_label_dim,
            num_phenos=self.num_phenos
        ).to(self.device)
        params = model.parameters()
        optimizer = optim.Adam(params, lr=self.init_lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_decay_gamma)
        return model, optimizer, lr_scheduler


    def _convert_input_data(self, data_tuple):
        # Unfolding
        x, nan_masks, fut_np, fut_mask_np, fut_avail_mask_np, tasks_np, tasks_mask_np, phenos_np, phenos_mask_np, towards, _, _, idpatients_np = data_tuple

        # Convert numpy to torch.tensor
        tasks = torch.from_numpy(tasks_np).long().to(self.device)
        tasks_mask = torch.from_numpy(tasks_mask_np * 1 + 1e-5).float().to(self.device)
        nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
        fut_mask = torch.from_numpy(fut_mask_np * 1 + 1e-5).float().to(self.device)
        fut_avail_mask = torch.from_numpy(fut_avail_mask_np.astype(int)).to(self.device)
        x, fut, towards = numpy2tensor(self.device,
                                x,
                                fut_np,
                                expand1darr(towards.astype(np.int64), 3, self.seq_dim)
                                )

        # Construct tuple
        input_data = (x, towards, fut_np, fut_mask_np, tasks_np, tasks_mask_np, idpatients_np, phenos_np, phenos_mask_np)
        input_info = (x, nan_masks, fut, fut_mask, fut_avail_mask, tasks, tasks_mask)
        return input_data, input_info


    def _get_entropy(self, log_density):
        N = log_density.shape[1]

        # max_arr is added for numerical stability
        max_arr, _ = torch.max(log_density, dim=1, keepdim=True)
        sum_arr = torch.sum(torch.exp(log_density - max_arr), dim=1)
        max_arr = max_arr.squeeze(1)
        entropy = max_arr + torch.log(sum_arr) - np.log(N * self.data_gen.num_rows)
        return entropy


    def _get_decomposed_kld(self, motion_z, motion_mu, motion_logvar, beta=5):
        # The decomposed KLD is split in 'Total correlation', 'Mutual information' and 'Dimension
        # wise KLD'. Like suggested in Chen et al. 2018 'Isolation sources of disentanglement in VAEs.
        N = motion_z.shape[0]
        K = motion_z.shape[1]

        logvar_expanded = motion_logvar.view(1, N, K)
        mu_expanded = motion_mu.view(1, N, K)
        samples_expanded = motion_z.view(N, 1, K)
        c_expanded = np.log(2 * np.pi) * torch.ones((N, 1, K)).cuda()

        # get log density assuming z is gaussian
        tmp = (samples_expanded - mu_expanded) * torch.exp(-0.5*logvar_expanded)
        log_density_z_j = -0.5 * (tmp * tmp + logvar_expanded + c_expanded)
        log_density_z = torch.sum(log_density_z_j, dim=2)

        # Get entropies
        marginal_entropy = self._get_entropy(log_density_z_j)
        joint_entropy = self._get_entropy(log_density_z)

        # Get nlogpz assuming prior N(0,1)
        nlogpz = -0.5 * (motion_z.pow(2) + np.log(2 * np.pi))

        # Get nlogqz_condx for the mutual information term
        tmp = (motion_z - motion_mu) * torch.exp(-0.5 * motion_logvar)
        nlogqz_condx = torch.sum(-0.5 * (tmp * tmp + motion_logvar + np.log(2 * np.pi)), dim=1)

        mutual_information = nlogqz_condx - joint_entropy
        total_correlation = joint_entropy - torch.sum(marginal_entropy, dim=1)
        dimwise_kld = torch.sum(marginal_entropy - nlogpz, dim=1)
        # Enable printing for debugging purposes.
        #print('joint entr: ', torch.mean(joint_entropy))
        #print('marginal entr: ', torch.mean(marginal_entropy))
        #print('nlogqz_condx: ', torch.mean(nlogqz_condx))
        #print('nlogpz: ', torch.mean(nlogpz))
        #print('Mutual Information: ', torch.mean(mutual_information))
        #print('Total Correlation: ', torch.mean(total_correlation))
        #print('Dimwise kld: ', torch.mean(dimwise_kld))
        #print('New KLD: ', torch.mean(mutual_information + total_correlation + dimwise_kld))

        decomposed_kld = torch.mean(mutual_information + beta * total_correlation + dimwise_kld)
        return decomposed_kld

    def loss_function(self, model_outputs, inputs_info):
        # Unfolding tuples
        x, nan_masks, fut, fut_mask, fut_avail_mask, tasks, tasks_mask = inputs_info
        recon_motion, pred_labels, recon_fut, pose_info, motion_info, phenos_info, task_latent = model_outputs
        pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar = pose_info
        motion_z, motion_mu, motion_logvar = motion_info
        phenos_pred, phenos_labels_np, pheno_latent = phenos_info

        # Posenet kld
        posenet_kld_multiplier = self._get_interval_multiplier(self.posenet_kld)
        posenet_kld_loss_indicator = -0.5 * torch.mean(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
        posenet_kld_loss = posenet_kld_multiplier * posenet_kld_loss_indicator

        # Motionnet kld
        motionnet_kld_multiplier = self._get_interval_multiplier(self.motionnet_kld)
        motionnet_kld_loss_indicator = -0.5 * torch.mean(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp())
        # For normal VAE: beta = 1, for beta-VAE set beta > 1
        beta = 1
        motionnet_kld_loss = motionnet_kld_multiplier * beta * motionnet_kld_loss_indicator
        # To use the TC-VAE uncommend the following 2 lines:
        # motion_decomposed_kld = self._get_decomposed_kld(motion_z, motion_mu, motion_logvar, beta=5)
        # motionnet_kld_loss = motionnet_kld_multiplier * 0.00025 * (motion_decomposed_kld)
        # Note by katja 15.4.20: The 0.00004 factor is rather arbitrary to get the decomposed kld on the same scale as the original one,
        # this depends on beta in motion_decomposed_kld. I haven't managed to find a clear relation to do this automatically.

        # Recon loss
        diff = x - recon_motion
        recon_loss_indicator = torch.mean(nan_masks * (diff ** 2))  # For evaluation
        recon_loss = self.recon_weight * recon_loss_indicator  # For error propagation

        # Future prediction loss
        diff = fut[fut_avail_mask == 1] - recon_fut[fut_avail_mask == 1]
        fut_predic_loss_indicator = torch.mean(fut_mask[fut_avail_mask == 1] * (diff ** 2))
        fut_predic_loss = self.fut_weight * fut_predic_loss_indicator

        # Latent recon loss
        squared_pose_z_seq = ((pose_z_seq - recon_pose_z_seq) ** 2)
        recon_latent_loss_indicator = torch.mean(squared_pose_z_seq)
        recon_latent_loss = 0 if self.latent_recon_loss is None else self.latent_recon_loss * recon_latent_loss_indicator

        # Gradient loss
        nan_mask_negibour_sum = self._calc_gradient_sum(nan_masks)
        gradient_mask = (nan_mask_negibour_sum.int() == 2).float()  # If the adjacent entries are both 1
        recon_grad_loss_indicator = torch.mean(gradient_mask * self._calc_gradient(recon_motion))
        pose_latent_grad_loss_indicator = torch.mean(self._calc_gradient(pose_z_seq))
        recon_grad_loss = self.recon_gradient * recon_grad_loss_indicator
        pose_latent_grad_loss = self.pose_latent_gradient * pose_latent_grad_loss_indicator

        # Classification loss
        class_loss_indicator, acc = self._get_classification_acc(pred_labels, tasks, tasks_mask)
        class_loss = self.classification_weight * class_loss_indicator

        # Identity classificaiton
        phenos_labels_tensor = torch.LongTensor(phenos_labels_np).to(self.device)
        phenos_loss_indicator, phenos_acc = self._phenos_criterion(phenos_pred, phenos_labels_tensor)
        phenos_loss = 0.001 * phenos_loss_indicator

        # Combine different losses
        ## KLD has to be set to 0 manually if it is turned off, otherwise it is not numerically stable
        motionnet_kld_loss = 0 if self.motionnet_kld is None else motionnet_kld_loss
        posenet_kld_loss = 0 if self.posenet_kld is None else posenet_kld_loss
        loss = recon_loss + posenet_kld_loss + motionnet_kld_loss + recon_grad_loss + pose_latent_grad_loss + \
               recon_latent_loss + class_loss + phenos_loss + fut_predic_loss

        return loss, (
            recon_loss_indicator, posenet_kld_loss_indicator, motionnet_kld_loss_indicator, recon_grad_loss_indicator,
            pose_latent_grad_loss_indicator, acc, phenos_loss_indicator, phenos_acc, fut_predic_loss_indicator)

    def _update_loss_meters(self, total_loss, indicators, train):

        recon, posekld, motionkld, recongrad, latentgrad, acc, phenos_loss, phenos_acc, fut_predic = indicators

        if train:
            self.loss_meter.update_meters(
                train_total_loss=total_loss.item(),
                train_recon=recon.item(),
                train_fut=fut_predic.item(),
                train_pose_kld=posekld.item(),
                train_motion_kld=motionkld.item(),
                train_recon_grad=recongrad.item(),
                train_latent_grad=latentgrad.item(),
                train_acc=acc,
                train_phenos_loss=phenos_loss.item(),
                train_phenos_acc=phenos_acc
            )
        else:
            self.loss_meter.update_meters(
                test_total_loss=total_loss.item(),
                test_recon=recon.item(),
                test_fut=fut_predic.item(),
                test_pose_kld=posekld.item(),
                test_motion_kld=motionkld.item(),
                test_recon_grad=recongrad.item(),
                test_latent_grad=latentgrad.item(),
                test_acc=acc,
                test_phenos_loss=phenos_loss.item(),
                test_phenos_acc=phenos_acc
            )

    def _print_for_each_iter(self, n_epochs, iter_idx, within_iter=True):
        # Print Info
        print("\rE %d/%d I %d/%d|Recon=%0.8f, %0.8f|KLD=%0.8f, %0.8f|Task=%0.3f, %0.3f|Phenos=%0.3f, %0.3f|Future=%0.8f, %0.8f" % (
            self.epoch,
            n_epochs,
            iter_idx,
            self.data_gen.num_rows / self.data_gen.m,
            self.loss_meter.get_meter_avg()["train_recon"],
            self.loss_meter.get_meter_avg()["test_recon"],
            self.loss_meter.get_meter_avg()["train_motion_kld"],
            self.loss_meter.get_meter_avg()["test_motion_kld"],
            self.loss_meter.get_meter_avg()["train_acc"],
            self.loss_meter.get_meter_avg()["test_acc"],
            self.loss_meter.get_meter_avg()["train_phenos_acc"],
            self.loss_meter.get_meter_avg()["test_phenos_acc"],
            self.loss_meter.get_meter_avg()["train_fut"],
            self.loss_meter.get_meter_avg()["test_fut"]
        ), flush=True, end=""
              )
        if not within_iter:
            print()

    def _phenos_criterion(self, phenos_pred_tensor, phenos_labels_tensor):
        # Loss function
        loss_indicator = torch.mean(self.class_criterion(phenos_pred_tensor, phenos_labels_tensor))

        # Calculate Accuracy
        phenos_pred_np = phenos_pred_tensor.cpu().detach().numpy()
        phenos_labels_np = phenos_labels_tensor.cpu().detach().numpy()
        phenos_acc = np.mean(np.argmax(phenos_pred_np, axis=1) == phenos_labels_np) * 100

        return loss_indicator, phenos_acc
