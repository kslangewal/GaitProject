import torch
import torch.nn as nn
import torch.optim as optim


def pose_block(input_channels,
               output_channels,
               dropout_p=0):
    LN_layer = nn.Linear(input_channels, output_channels)
    bn_layer = nn.BatchNorm1d(output_channels, track_running_stats=False)
    relu_layer = nn.ReLU()
    droput_layer = nn.Dropout(dropout_p)
    block_list = [
        LN_layer,
        bn_layer,
        relu_layer,
        droput_layer
    ]
    return block_list


def motion_encoding_block(input_channels,
                          output_channels,
                          kernel_size,
                          stride,
                          dropout_p=0.25):
    conv_layer = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
    bn_1_layer = nn.BatchNorm1d(output_channels, track_running_stats=False)
    selu_1_layer = nn.SELU()
    droput_1_layer = nn.Dropout(dropout_p)
    fc_layer = nn.Conv1d(output_channels, output_channels, kernel_size=1)
    bn_2_layer = nn.BatchNorm1d(output_channels, track_running_stats=False)
    selu_2_layer = nn.SELU()
    droput_2_layer = nn.Dropout(dropout_p)
    block_list = [
        conv_layer,
        bn_1_layer,
        selu_1_layer,
        droput_1_layer,
        fc_layer,
        bn_2_layer,
        selu_2_layer,
        droput_2_layer
    ]
    return block_list


def motion_decoding_block(input_channels, output_channels, kernel_size, stride):
    conv_layer = nn.ConvTranspose1d(input_channels, output_channels,
                                    kernel_size=kernel_size,
                                    stride=stride)
    bn_1_layer = nn.BatchNorm1d(output_channels, track_running_stats=False)
    selu_1_layer = nn.SELU()
    fc_layer = nn.ConvTranspose1d(output_channels, output_channels, kernel_size=1)
    bn_2_layer = nn.BatchNorm1d(output_channels, track_running_stats=False)
    selu_2_layer = nn.SELU()
    block_list = [
        conv_layer,
        bn_1_layer,
        selu_1_layer,
        fc_layer,
        bn_2_layer,
        selu_2_layer
    ]
    return block_list


class LshapeCounter:
    def __init__(self, L_in):
        self.L = L_in

    def updateL(self, kernel_size, dilation=1, padding=0, stride=1, decode=False):
        if decode is False:
            self.L = self.calc_conv1d_Lout(self.L, kernel_size, dilation, padding, stride)
        else:
            self.L = self.calc_deconv1d_Lout(self.L, kernel_size, dilation, padding, stride)
        return self.L

    @staticmethod
    def calc_conv1d_Lout(Lin, kernel_size, dilation=1, padding=0, stride=1):
        Lout = ((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        return Lout

    @staticmethod
    def calc_deconv1d_Lout(Lin, kernel_size, dilation=1, padding=0, stride=1, output_padding=0):
        Lout = (Lin - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        return Lout


class Reparameterize(nn.Module):
    def __init__(self, device, latent_dim):
        super(Reparameterize, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

    def forward(self, h):
        # Reparameterize
        mu, logvar = h[:, 0:self.latent_dim], h[:, self.latent_dim:]
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z, mu, logvar


class Flatten(nn.Module):
    """
    Convert tensor from shape (a, b, c) to (a * b, c)
    """

    def forward(self, x):
        return x.reshape(x.size(0) * x.size(1), x.size(2))


class UnFlatten(nn.Module):
    """
    Convert tensor from shape (a * b, c) to (a, b, c)
    """

    def __init__(self, b):
        super(UnFlatten, self).__init__()
        self.b = b

    def forward(self, x):
        return x.reshape(-1, self.b, x.size(1))


class FlattenLastDim(nn.Module):
    """
    Convert tensor from shape (a, b, c) to (a, b * c)
    """

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Unsqueeze(nn.Module):
    """
    Convert tensor from shape (a, b) to (a, b, 1)
    """

    def forward(self, x):
        return x.reshape(x.size(0), x.size(1), 1)


class Transpose(nn.Module):
    """
    Convert tensor from shape (a, b, c) to (a, c, b)
    """

    def forward(self, x):
        return x.permute(0, 2, 1)


class SpatioTemporalVAE(nn.Module):
    def __init__(self,
                 fea_dim=50,
                 seq_dim=128,
                 fut_dim=16,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=True,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=True,
                 futnet_hidden_dim=512,
                 device=None
                 ):
        """
        This network takes input with shape (m, fea_dim, seq_dim), and reconstructs it, where m is number of samples.
        This network also does classification with the motion's latents. Number of classes is hard-coded as 8 (see below)

        New 31.03.2020: the network also predicts 'fut_dim' frames following the sequence.

        Parameters
        ----------
        fea_dim : int
        seq_dim : int
        fut_dim : int
        posenet_latent_dim : int
        posenet_dropout_p : float
        posenet_kld : bool
        motionnet_latent_dim : int
        motionnet_hidden_dim : int
        motionnet_dropout_p : float
        motionnet_kld : bool
        futnet_hidden_dim : int
        """
        # # Loading parameters
        # Data dimension
        self.fea_dim = fea_dim
        self.seq_dim = seq_dim
        self.fut_dim = fut_dim
        self.n_classes = 8
        # Autoencoder for single pose
        self.posenet_latent_dim = posenet_latent_dim
        self.posenet_dropout_p = posenet_dropout_p
        self.posenet_kld = posenet_kld
        # Autoencoder for a sequence of poses (motion)
        self.motionnet_latent_dim = motionnet_latent_dim
        self.motionnet_hidden_dim = motionnet_hidden_dim
        self.motionnet_dropout_p = motionnet_dropout_p
        self.motionnet_kld = motionnet_kld
        # Future prediction
        self.futnet_hidden_dim = futnet_hidden_dim
        # Others
        super(SpatioTemporalVAE, self).__init__()
        self.device = torch.device('cuda:0') if device is None else device

        # # Initializing network architecture
        self.transpose_layer = Transpose()
        self.flatten_layer = Flatten()
        self.unflatten_layer = UnFlatten(self.seq_dim)
        self.pose_vae = PoseVAE(fea_dim=self.fea_dim,
                                latent_dim=self.posenet_latent_dim,
                                kld=self.posenet_kld,
                                dropout_p=self.posenet_kld,
                                device=self.device)

        self.motion_vae = MotionVAE(fea_dim=self.posenet_latent_dim,
                                    seq_dim=self.seq_dim,
                                    hidden_dim=self.motionnet_hidden_dim,
                                    latent_dim=self.motionnet_latent_dim,
                                    kld=self.motionnet_kld,
                                    dropout_p=self.motionnet_dropout_p,
                                    device=self.device)

        self.class_net = TaskNet(input_dim=self.motionnet_latent_dim,
                                 n_classes=self.n_classes,
                                 device=self.device)

        self.fut_net = FutureNet(fut_dim=self.fut_dim,
                                 fea_dim=self.fea_dim,
                                 z_latent_dim=self.motionnet_latent_dim,
                                 p_latent_dim=self.posenet_latent_dim,
                                 hidden_dim=self.futnet_hidden_dim,
                                 dropout_p=self.posenet_dropout_p,
                                 device=self.device)

    def forward(self, *input):
        x, _, _ = input
        (pose_z_seq, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar) = self.encode(x)
        recon_motion, recon_pose_z_seq = self.decode(motion_z)  # Convert (m, motion_latent_dim) to (m, fea, seq)
        pred_labels, task_latent = self.class_net(motion_z)  # Convert (m, motion_latent_dim) to (m, n_classes)
        fut_recon = self.fut_net(motion_z)
        return recon_motion, pred_labels, fut_recon, (pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar), (
        motion_z, motion_mu, motion_logvar), task_latent

    def encode(self, x):
        out = self.transpose_flatten(x)  # Convert (m, fea, seq) to (m * seq, fea)
        pose_out = self.pose_encode(out)  # Convert (m * seq, fea) to (m * seq, pose_latent_dim (or *2 if kld=True) )
        pose_z, pose_mu, pose_logvar = self.pose_bottoleneck(pose_out)  # all outputs (m * seq, pose_latent_dim)
        pose_z_seq = self.unflatten_transpose(pose_z)  # Convert (m * seq, pose_latent_dim) to (m, pose_latent_dim, seq)
        out = self.motion_encode(
            pose_z_seq)  # Convert (m, pose_latent_dim, seq) to (m, motion_latent_dim (or *2 if kld=True) )
        motion_z, motion_mu, motion_logvar = self.motion_bottoleneck(out)  # all outputs (m, motion_latent_dim)
        return (pose_z_seq, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar)

    def decode(self, motion_z):
        recon_pose_z_seq = self.motion_decode(motion_z)  # Convert (m, motion_latent_dim) to  (m, pose_latent_dim, seq)
        out = self.transpose_flatten(
            recon_pose_z_seq)  # Convert (m, pose_latent_dim, seq) to (m * seq, pose_latent_dim)
        out = self.pose_decode(out)  # Convert (m * seq, pose_latent_dim) to (m * seq, fea)
        recon_motion = self.unflatten_transpose(out)  # Convert (m * seq, fea) to (m, fea, seq)
        return recon_motion, recon_pose_z_seq

    def pose_encode(self, x):
        out = self.pose_vae.encode(x)
        return out

    def pose_bottoleneck(self, x):
        if self.posenet_kld:
            z, mu, logvar = self.pose_vae.pose_reparams(x)
        else:
            z = x
            mu = z.clone()  # dummy assignment
            logvar = mu  # dummy assignment

        return z, mu, logvar

    def pose_decode(self, x):
        out = self.pose_vae.decode(x)
        return out

    def motion_encode(self, x):
        out = self.motion_vae.encode(x)
        return out

    def motion_bottoleneck(self, x):
        if self.motionnet_kld:
            z, mu, logvar = self.motion_vae.motion_reparams(x)
        else:
            z = x.clone()
            mu, logvar = z, z  # dummy assignment
        return z, mu, logvar

    def motion_decode(self, x):
        out = self.motion_vae.decode(x)
        return out

    def transpose_flatten(self, x):
        out = self.transpose_layer(x)  # Convert (m, fea, seq) to (m, seq, fea)
        out = self.flatten_layer(out)  # Convert (m, seq, fea) to (m * seq, fea)
        return out

    def unflatten_transpose(self, x):
        out = self.unflatten_layer(x)  # Convert (m * seq, fea) to (m, seq, fea)
        out = self.transpose_layer(out)  # Convert (m, seq, fea) to (m, fea, seq)
        return out


class PoseVAE(nn.Module):

    def __init__(self, fea_dim, latent_dim, kld, dropout_p, device=None):
        """
        PoseVAE takes in data with shape (m, input_dims), and reconstructs it with bottleneck layer (m, latent_dims),
        where m is number of samples.

        Parameters
        ----------
        fea_dim : int
        latent_dim : int
            A value smaller than the dimension of the last layer (Default=32) before bottleneck layer.
        kld : bool
            Stochastic sampling if True, deterministic if False
        dropout_p : int
        """

        # Model setting
        super(PoseVAE, self).__init__()
        self.kld = kld
        self.fea_dim, self.latent_dim = fea_dim, latent_dim
        self.connecting_dim = self.latent_dim * 2 if self.kld else self.latent_dim
        self.encode_units = [512, 128, 64, 32]
        self.decode_units = [32, 64, 128, 512]
        self.device = torch.device('cuda:0') if device is None else device

        # Encoder

        self.en_blk1 = nn.Sequential(*pose_block(input_channels=self.fea_dim,
                                                 output_channels=self.encode_units[1],
                                                 dropout_p=dropout_p))
        self.en_blk2 = nn.Sequential(*pose_block(input_channels=self.encode_units[1],
                                                 output_channels=self.encode_units[2],
                                                 dropout_p=dropout_p))
        self.en_blk3 = nn.Sequential(*pose_block(input_channels=self.encode_units[2],
                                                 output_channels=self.encode_units[3],
                                                 dropout_p=dropout_p))
        self.en2latents = nn.Linear(self.encode_units[3], self.connecting_dim)

        self.pose_reparams = Reparameterize(self.device, self.latent_dim)

        # Decode
        self.latents2de = nn.Linear(self.latent_dim, self.decode_units[0])

        self.de_blk1 = nn.Sequential(*pose_block(input_channels=self.decode_units[0],
                                                 output_channels=self.decode_units[1],
                                                 dropout_p=dropout_p))
        self.de_blk2 = nn.Sequential(*pose_block(input_channels=self.decode_units[1],
                                                 output_channels=self.decode_units[2],
                                                 dropout_p=dropout_p))
        self.de_blk3 = nn.Sequential(*pose_block(input_channels=self.decode_units[2],
                                                 output_channels=self.decode_units[3],
                                                 dropout_p=dropout_p))
        self.final_layer = nn.Linear(self.decode_units[3], self.fea_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x : pytorch.tensor

        Returns
        -------
        out : pytorch.tensor
        mu : pytorch.tensor
        logvar : pytorch.tensor
        z : pytorch.tensor
        """
        # Encoder
        out = self.encode(x)

        if self.kld:
            # Sampling from latents and concatenate with labels
            z, mu, logvar = self.pose_reparams(out)
        else:
            z = out
            mu = z.clone()  # dummy assignment
            logvar = z  # dummy assignment
        out = self.decode(z)
        return out, mu, logvar, z

    def encode(self, x):

        out = self.en_blk1(x)

        out = self.en_blk2(out)

        out = self.en_blk3(out)

        out = self.en2latents(out)
        return out

    def decode(self, z):

        out = self.latents2de(z)

        out = self.de_blk1(out)

        out = self.de_blk2(out)

        out = self.de_blk3(out)

        out = self.final_layer(out)
        return out


class MotionVAE(nn.Module):
    def __init__(self, fea_dim=50, seq_dim=128, hidden_dim=1024, latent_dim=8, kld=False, dropout_p=0, device=None):
        """
        Temporal Variational Autoencoder (TemporalVAE)
        In Gait analysis. we want a VAE function f(x) that follows the shapes:

            x = (N, C, L), where N = batch_size, C = n_features + n_labels, L = sequence length (time), specifically,
            C = 25*2+8.

            f(x) = (N, C_2, L), where C_2 = 25*2, denoting the x, y-coordinates of the 25 keypoints.

        Parameters
        ----------
        fea_dim : int
            Number of channels (C). In gait analysis, n_channels = 25 * 2 + 8, which is (n_features + n_labels).
        seq_dim : int
            The length of input sequence. In gait analysis, it is the sequence length.
        """

        # Init
        super(MotionVAE, self).__init__()
        self.fea_dim, self.seq_dim, self.latent_dim = fea_dim, seq_dim, latent_dim
        self.kld = kld
        self.connecting_dim = self.latent_dim * 2 if self.kld else self.latent_dim
        self.device = torch.device('cuda:0') if device is None else device

        # Record the time dimension
        self.L_encode_counter = LshapeCounter(seq_dim)
        self.encoding_kernels = [5, 5, 5, 5, 5]
        self.encoding_strides = [1, 2, 2, 2, 2]
        self.decoding_kernels = [5, 5, 5, 5, 8]
        self.decoding_strides = [1, 2, 2, 2, 2]
        self.Ls_encode = [self.L_encode_counter.updateL(kernel_size=x, stride=y) for x, y, in
                          zip(self.encoding_kernels, self.encoding_strides)]

        # Encoder
        self.first_layer = nn.Conv1d(self.fea_dim,
                                     hidden_dim,
                                     kernel_size=self.encoding_kernels[0],
                                     stride=self.encoding_strides[0])
        self.en_blk1 = nn.Sequential(*motion_encoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.encoding_kernels[1],
                                                            stride=self.encoding_strides[1],
                                                            dropout_p=dropout_p))
        self.en_blk2 = nn.Sequential(*motion_encoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.encoding_kernels[2],
                                                            stride=self.encoding_strides[2],
                                                            dropout_p=dropout_p))
        self.en_blk3 = nn.Sequential(*motion_encoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.encoding_kernels[3],
                                                            stride=self.encoding_strides[3],
                                                            dropout_p=dropout_p))
        self.en_blk4 = nn.Sequential(*motion_encoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.encoding_kernels[4],
                                                            stride=self.encoding_strides[4],
                                                            dropout_p=dropout_p))
        self.en2latents = nn.Sequential(
            FlattenLastDim(),
            nn.Linear(hidden_dim * int(self.Ls_encode[4]), self.connecting_dim)
        )

        self.motion_reparams = Reparameterize(self.device, self.latent_dim)

        self.latents2de = nn.Sequential(
            Unsqueeze(),
            nn.ConvTranspose1d(self.latent_dim,
                               hidden_dim,
                               kernel_size=self.decoding_kernels[0],
                               stride=self.decoding_strides[0])
        )

        self.de_blk1 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[1],
                                                            stride=self.decoding_strides[1]))
        self.de_blk2 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[2],
                                                            stride=self.decoding_strides[2]))
        self.de_blk3 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[3],
                                                            stride=self.decoding_strides[3]))
        self.de_blk4 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[4],
                                                            stride=self.decoding_strides[4]))
        self.final_layer = nn.Conv1d(hidden_dim, self.fea_dim, kernel_size=1)

    def forward(self, x):
        # Encoder
        out = self.encode(x)

        if self.kld:
            z, mu, logvar = self.motion_reparams(out)
        else:
            z = out
            mu = z.clone()  # dummy assignment
            logvar = mu  # dummy assignment

        out = self.decode(z)
        return out, mu, logvar, z

    def encode(self, x):

        out = self.first_layer(x)

        out = self.en_blk1(out)

        out = self.en_blk2(out)

        out = self.en_blk3(out)

        out = self.en_blk4(out)

        out = self.en2latents(out)
        return out

    def decode(self, z):

        out = self.latents2de(z)

        out = self.de_blk1(out)

        out = self.de_blk2(out)

        out = self.de_blk3(out)

        out = self.de_blk4(out)

        out = self.final_layer(out)

        return out


class TaskNet(nn.Module):

    def __init__(self, input_dim, n_classes, device=None):
        """
        PoseVAE takes in data with shape (m, input_dims), and reconstructs it with bottleneck layer (m, latent_dims),
        where m is number of samples.

        Parameters
        ----------
        input_dim : int
        n_classes : int
            Also the dimension of final layer
        """

        # Model setting
        super(TaskNet, self).__init__()
        self.input_dim, self.n_classes = input_dim, n_classes
        self.device = torch.device('cuda:0') if device is None else device
        self.encode_units = [128, 64, 32, 16]

        # Encoder
        self.first_layer = nn.Linear(self.input_dim, self.encode_units[0])

        self.en_blk1 = nn.Sequential(*pose_block(input_channels=self.encode_units[0],
                                                 output_channels=self.encode_units[1]
                                                 ))
        self.en_blk2 = nn.Sequential(*pose_block(input_channels=self.encode_units[1],
                                                 output_channels=self.encode_units[2]
                                                 ))
        self.en_blk3 = nn.Sequential(*pose_block(input_channels=self.encode_units[2],
                                                 output_channels=self.encode_units[3]
                                                 ))
        self.final_layer = nn.Linear(self.encode_units[3], self.n_classes)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        task_latent = self.encode(x)
        out = self.sigmoid_layer(task_latent)
        return out, task_latent

    def encode(self, x):

        out = self.first_layer(x)

        out = self.en_blk1(out) + out[:, 0:int(self.encode_units[1])]

        out = self.en_blk2(out) + out[:, 0:int(self.encode_units[2])]

        out = self.en_blk3(out) + out[:, 0:int(self.encode_units[3])]

        out = self.final_layer(out)
        return out


class FutureNet(nn.Module):
    def __init__(self, fut_dim, fea_dim, z_latent_dim, p_latent_dim, hidden_dim, dropout_p, device=None):
        """
        FutureNet takes in the latent dimension of the VAE of size (m, z_latent_dim) and converts this into a sequence
        of (m, fea_dim, fut_dim), representing the future fut_dim frames of the sequence.

        Inputs:
        fut_dim: int
        fea_dim: int
        z_latent_dim: int
        p_latent_dim: int
        hidden_dim: int
        dropout_p: int
        """
        # Model setting
        super(FutureNet, self).__init__()
        self.fut_dim = fut_dim
        self.z_latent_dim = z_latent_dim
        self.device = torch.device('cuda:0') if device is None else device

        self.decoding_kernels = [5, 5, 5, 4]
        self.decoding_strides = [1, 1, 1, 1]
        self.decode_units = [32, 64, 128, 512]

        # Layer set-up
        self.transpose_layer = Transpose()
        self.flatten_layer = Flatten()
        self.unflatten_layer = UnFlatten(self.fut_dim)

        self.latents2de = nn.Sequential(
            Unsqueeze(),
            nn.ConvTranspose1d(z_latent_dim,
                               hidden_dim,
                               kernel_size=self.decoding_kernels[0],
                               stride=self.decoding_strides[0]))

        self.de_blk1 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[1],
                                                            stride=self.decoding_strides[1]))

        self.de_blk2 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[2],
                                                            stride=self.decoding_strides[2]))

        self.de_blk3 = nn.Sequential(*motion_decoding_block(hidden_dim,
                                                            hidden_dim,
                                                            kernel_size=self.decoding_kernels[3],
                                                            stride=self.decoding_strides[3]))

        self.middle_layer = nn.Conv1d(hidden_dim, p_latent_dim, kernel_size=1)

        self.platents2de = nn.Linear(p_latent_dim, self.decode_units[0])

        self.pde_blk1 = nn.Sequential(*pose_block(input_channels=self.decode_units[0],
                                                    output_channels=self.decode_units[1],
                                                    dropout_p=dropout_p))

        self.pde_blk2 = nn.Sequential(*pose_block(input_channels=self.decode_units[1],
                                                    output_channels=self.decode_units[2],
                                                    dropout_p=dropout_p))

        self.pde_blk3 = nn.Sequential(*pose_block(input_channels=self.decode_units[2],
                                                    output_channels=self.decode_units[3],
                                                    dropout_p=dropout_p))

        self.final_layer = nn.Linear(self.decode_units[3], fea_dim)

    def forward(self, motion_z):
        out = self.latents2de(motion_z)
        out = self.de_blk1(out)
        out = self.de_blk2(out)
        out = self.de_blk3(out)
        out = self.middle_layer(out)

        out = self.transpose_flatten(out)

        out = self.platents2de(out)
        out = self.pde_blk1(out)
        out = self.pde_blk2(out)
        out = self.pde_blk3(out)
        out = self.final_layer(out)

        out = self.unflatten_transpose(out)

        return out

    def transpose_flatten(self, x):
        out = self.transpose_layer(x)
        out = self.flatten_layer(out)
        return out

    def unflatten_transpose(self, x):
        out = self.unflatten_layer(x)
        out = self.transpose_layer(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0')

    network_params = {
        "posenet": {
            "latent_dim": 2,
            "dropout_p": 0.25,
            "kld": False
        },
        "motionnet": {
            "latent_dim": 2,
            "dropout_p": 0.25,
            "hidden_dim": 512,
            "kld": False
        }
    }

    combined_model = SpatioTemporalVAE(network_params=network_params,
                                       fea_dim=50,
                                       seq_dim=128).to(device)

    sample_pose = torch.rand(size=(30 * 128, 50)).float().to(device)
    sample_motion = torch.rand(size=(30, 50, 128)).float().to(device)

    combined_params = combined_model.parameters()

    params = list(combined_params)
    optimizer = optim.Adam(params, lr=0.001)
    for i in range(50):
        optimizer.zero_grad()
        recon_motion, (pose_z, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar) = combined_model(
            sample_motion)
        loss = torch.mean((sample_motion - recon_motion) ** 2)

        print(loss)
        loss.backward()
        optimizer.step()
