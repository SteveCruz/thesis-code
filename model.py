##############################################################################################################################################################
##############################################################################################################################################################
"""
Autoencoder model definitions are located inside this script.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import math
import torch
from torch import nn
from pytorch_msssim import SSIM, MS_SSIM
from pytorch_metric_learning import losses
import lpips
import torchvision

##############################################################################################################################################################
##############################################################################################################################################################

def get_activation_function(model_config):

    # from the torch.nn module, get the activation function specified in the config. 
    # make sure the naming is correctly spelled according to the torch name
    if hasattr(nn, model_config["activation"]):
        activation = getattr(nn, model_config["activation"])
        return activation()
    else:
        raise ValueError("Activation function does not exist.")

##############################################################################################################################################################

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # As an example, we print the shape here.
        print(x.shape)
        return x

##############################################################################################################################################################

def create_ae_model(config):

    # define the autoencoder model
    if config["model"]["type"] == "fc_ae":
        return FCAE(config)
    elif config["model"]["type"] == "conv_ae":
        return ConvAE(config)
    elif config["model"]["type"] == "conv_vae":
        return ConvVAE(config)
    elif config["model"]["type"] == "dropout_fc_ae":
        return DropoutFCAE(config)
    elif config["model"]["type"] == "dropout_conv_ae":
        return DropoutConvAE(config)
    elif config["model"]["type"] == "mlp":
        return MLP(config)
    elif config["model"]["type"] == "conv":
        return Conv(config)
    elif config["model"]["type"] == "extractor":
        return ExtractorAE(config)
    elif config["model"]["type"] == "multi_channel_ae":
        return MultiChannelAE(config)
    elif config["model"]["type"] == "multi_channel_extractor_ae":
        return MultiChannelExtractorAE(config)

##############################################################################################################################################################

class BaseAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for ease of use, get the different configs
        self.model_config, self.data_config, self.training_config = config["model"], config["dataset"], config["training"]

        # some definitions
        self.type =  self.model_config["type"]
        self.nbr_input_channels = 3 if self.data_config["name"].lower() == "mpi3d" else 1

        # latent space dimension
        self.latent_dim = self.model_config["dimension"]

        # define the activation function to use
        self.activation = get_activation_function(self.model_config)      

        # define the loss function to use
        self.which_recon_loss = self.training_config["loss"]
        self.which_metric_loss = self.model_config["metric"]
        self._init_loss()

    def print_model(self):

        print("=" * 57)
        print("Loss function used: ", self.criterion)
        print("Metric function used: ", self._metric_loss_fn)
        print("=" * 57)
        print("The autoencoder is defined as: ")
        print("=" * 57)
        print(self)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)

    def _init_recon_loss(self):

        # define the loss function to use
        if self.which_recon_loss == "MSE":
            self.criterion = nn.MSELoss(reduction="sum")
        elif self.which_recon_loss == "L1":     
            self.criterion = nn.L1Loss(reduction="sum")
        elif self.which_recon_loss == "BCE":     
            self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
        elif self.which_recon_loss == "Huber":     
            self.criterion = nn.HuberLoss(reduction="sum", delta=1.0)
        elif self.which_recon_loss == "SSIM":     
            self.criterion = SSIM(data_range=1.0, size_average=True, channel=self.nbr_input_channels)
        elif self.which_recon_loss == "MSSSIM":     
            self.criterion = MS_SSIM(data_range=1.0, size_average=True, channel=self.nbr_input_channels, win_size=7)
        elif self.which_recon_loss == "Perceptual":     
            self.criterion = lpips.LPIPS(net='vgg', lpips=False)
            self.criterion_preprocess = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif self.which_recon_loss == "LPIPS":     
            self.criterion = lpips.LPIPS(net='vgg', lpips=True)
            self.criterion_preprocess = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            raise ValueError("Loss definition does not exist.")

    def _init_metric_loss(self):

        # init metric  loss
        if self.which_metric_loss == "":
            self._metric_loss_fn = None
        elif self.which_metric_loss == "triplet":
            self._metric_loss_fn = losses.TripletMarginLoss(margin=0.2, swap=True)
        elif self.which_metric_loss == "angular":
            self._metric_loss_fn = losses.AngularLoss()
        elif self.which_metric_loss == "npair":
            self._metric_loss_fn = losses.NPairsLoss()
        elif self.which_metric_loss == "ntxent":
            self._metric_loss_fn = losses.NTXentLoss()
        elif self.which_metric_loss == "contrastive":
            self._metric_loss_fn = losses.ContrastiveLoss()
        else:
            raise ValueError("Metric function does not exist.")

    def _init_loss(self):

        # define both losses
        self._init_recon_loss()
        self._init_metric_loss()

    def metric_loss(self, embedding, labels) : 
        return self._metric_loss_fn(embedding, labels)

    def loss(self, prediction, target):

        if self.which_recon_loss in ["SSIM", "MSSSIM"]:
            return 1-self.criterion(prediction, target)
        elif self.which_recon_loss == "BCE":   
            # for BCE its better not to mean over batch dimension
            return self.criterion(prediction, target) 
        elif self.which_recon_loss in ["Perceptual", "LPIPS"]:   
            if prediction.shape[1] != 3:
                prediction = prediction.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
            # map images into range [-1, 1]
            prediction = self.criterion_preprocess(prediction)
            target = self.criterion_preprocess(target)
            return self.criterion.forward(prediction, target).mean()
        else:
            # sum over the pixel dimension and divide over batch dimension
            # this is better than the mean over all pixels
            return self.criterion(prediction, target) / target.shape[0]

    def _enable_dropout(self):
        if self.dropout_rate == 0:
            raise ValueError("Enabling dropout does not make any sense.")
        for each_module in self.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, mu):
        return self.decoder(mu)

##############################################################################################################################################################

class FCAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        self.pixel_height = self.data_config["img_size"]
        self.pixel_width = self.data_config["img_size"]

        # width of all layers
        # scale with dropout rate for a fair comparison
        self.dropout_rate = self.model_config["dropout"] 
        self.layer_width = int(self.model_config["width"] * (1+self.dropout_rate))

        self.encoder = nn.Sequential()
        self.encoder.add_module("fc_1", nn.Linear(in_features=self.pixel_width*self.pixel_height, out_features=self.layer_width, bias=True))
        self.encoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_2", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.encoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_3", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.encoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_4", nn.Linear(in_features=self.layer_width, out_features=self.latent_dim, bias=True))

        self.decoder = nn.Sequential()
        self.decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.layer_width, bias=True))
        self.decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("fc_2", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("fc_3", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("fc_4", nn.Linear(in_features=self.layer_width, out_features=self.pixel_height*self.pixel_width, bias=True))

    def forward(self, x):

        # do the reshaping outside the model definition
        # otherwise the jacobian for the reshape layers does not work
        x = x.reshape(-1, self.pixel_height*self.pixel_width*self.nbr_input_channels)
        mu = self.encode(x)
        z = self.decode(mu)
        z = z.reshape(-1, self.nbr_input_channels, self.pixel_height, self.pixel_width)

        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class ConvAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        # get the dropout rate
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.channels_before_fc = math.ceil(64 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels * self.img_size_scale**2

        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True))

        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc, bias=True))
        self.decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))

    def forward(self, x):

        # do the reshaping outside the model definition
        # otherwise the jacobian for the reshape layers does not work
        mu = self.encode(x)
        z = self.decode(mu)

        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class DropoutFCAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        self.pixel_height = self.data_config["img_size"]
        self.pixel_width = self.data_config["img_size"]

        # width of all layers
        # scale with dropout rate for a fair comparison
        self.dropout_rate = self.model_config["dropout"] 
        self.layer_width = int(self.model_config["width"] * (1+self.dropout_rate))

        # encoder
        self.encoder_first = nn.Sequential()
        self.encoder_first.add_module("fc_1", nn.Linear(in_features=self.pixel_width*self.pixel_height, out_features=self.layer_width, bias=True))
        self.encoder_first.add_module("activation_1", self.activation)

        self.encoder_second = nn.Sequential()
        self.encoder_second.add_module("fc_2", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.encoder_second.add_module("activation_2", self.activation)

        self.encoder_third = nn.Sequential()
        self.encoder_third.add_module("fc_3", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.encoder_third.add_module("activation_3", self.activation)

        self.encoder_final = nn.Linear(in_features=self.layer_width, out_features=self.latent_dim, bias=True)

        # decoder
        self.decoder_first = nn.Sequential()
        self.decoder_first.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.layer_width, bias=True))
        self.decoder_first.add_module("activation_1", self.activation)

        self.decoder_second = nn.Sequential()
        self.decoder_second.add_module("fc_2", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.decoder_second.add_module("activation_2", self.activation)

        self.decoder_third = nn.Sequential()
        self.decoder_third.add_module("fc_3", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.decoder_third.add_module("activation_3", self.activation)

        self.decoder_final = nn.Linear(in_features=self.layer_width, out_features=self.pixel_height*self.pixel_width, bias=True)


    def _get_binary_mask(self, batch_size, device):
        # https://discuss.pytorch.org/t/how-to-fix-the-dropout-mask-for-different-batch/7119
        # Please note that the Bernoulli distribution samples 0 with the probability (1-p), 
        # contrary to dropout implementations, which sample 0 with probability p.
        return torch.bernoulli(torch.full([batch_size , self.layer_width], 1-self.dropout_rate, device=device, requires_grad=False))/(1-self.dropout_rate)

    def _define_and_fix_new_dropout_mask(self, x):

        self.encoder_first_mask = self._get_binary_mask(x.shape[0], device=x.device)
        self.encoder_second_mask = self._get_binary_mask(x.shape[0], device=x.device)
        self.encoder_third_mask = self._get_binary_mask(x.shape[0], device=x.device)

        self.decoder_first_mask = self._get_binary_mask(x.shape[0], device=x.device)
        self.decoder_second_mask = self._get_binary_mask(x.shape[0], device=x.device)
        self.decoder_third_mask = self._get_binary_mask(x.shape[0], device=x.device)

    def encoding(self, x, random):

        x = self.encoder_first(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_first_mask 

        x = self.encoder_second(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_second_mask 
            

        x = self.encoder_third(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_third_mask 
            
        x = self.encoder_final(x)

        return x

    def decoding(self, x, random):

        x = self.decoder_first(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_first_mask 

        x = self.decoder_second(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_second_mask 

        x = self.decoder_third(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_third_mask 
        
        x = self.decoder_final(x)

        return x

    def forward(self, x, random=True):

        x = x.reshape(-1, self.pixel_height*self.pixel_width*self.nbr_input_channels)
        mu = self.encoding(x, random)
        xhat = self.decoding(mu, random)
        xhat = xhat.reshape(-1, self.nbr_input_channels, self.pixel_height, self.pixel_width)

        return {"xhat":xhat, "mu": mu}

##############################################################################################################################################################

class DropoutConvAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        # get the dropout rata
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.channels_before_fc = math.ceil(64 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))
        
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels * self.img_size_scale**2


        # encoder
        self.encoder_first = nn.Sequential()
        self.encoder_first.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_first.add_module("activation_1", self.activation)

        self.encoder_second = nn.Sequential()
        self.encoder_second.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_second.add_module("activation_2", self.activation)

        self.encoder_third = nn.Sequential()
        self.encoder_third.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_third.add_module("activation_3", self.activation)

        self.encoder_fourth = nn.Sequential()
        self.encoder_fourth.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder_fourth.add_module("activation_4", self.activation)

        self.encoder_fifth = nn.Sequential()
        self.encoder_fifth.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder_fifth.add_module("activation_5", self.activation)

        self.encoder_final = nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True)

        self.flatten = nn.Flatten()

        # decoder
        self.decoder_first = nn.Sequential()
        self.decoder_first.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.decoder_first.add_module("activation_1", self.activation)

        self.decoder_second = nn.Sequential()
        self.decoder_second.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc, bias=True))
        self.decoder_second.add_module("activation_2", self.activation)

        self.decoder_third = nn.Sequential()
        self.decoder_third.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder_third.add_module("activation_3", self.activation)

        self.decoder_fourth = nn.Sequential()
        self.decoder_fourth.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder_fourth.add_module("activation_4", self.activation)

        self.decoder_fifth = nn.Sequential()
        self.decoder_fifth.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder_fifth.add_module("activation_5", self.activation)

        self.decoder_final = nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2)

        self.unflatten = nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale))


    def _get_binary_mask(self, shape, device):
        # https://discuss.pytorch.org/t/how-to-fix-the-dropout-mask-for-different-batch/7119
        # Please note that the Bernoulli distribution samples 0 with the probability (1-p), 
        # contrary to dropout implementations, which sample 0 with probability p.
        if len(shape) == 4:
            return torch.bernoulli(torch.full([shape[0], shape[1], shape[2], shape[3]], 1-self.dropout_rate, device=device, requires_grad=False))/(1-self.dropout_rate)
        else:
            return torch.bernoulli(torch.full([shape[0], shape[1]], 1-self.dropout_rate, device=device, requires_grad=False))/(1-self.dropout_rate)

    def _define_and_fix_new_dropout_mask(self, x):

        self.encoder_first_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 8*self.reshape_channels, 8*self.reshape_channels], device=x.device)
        self.encoder_second_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 4*self.reshape_channels, 4*self.reshape_channels], device=x.device)
        self.encoder_third_mask = self._get_binary_mask([x.shape[0], 2*self.base_number_chnanels, 2*self.reshape_channels, 2*self.reshape_channels], device=x.device)
        self.encoder_fourth_mask = self._get_binary_mask([x.shape[0], 2*self.base_number_chnanels, self.reshape_channels, self.reshape_channels], device=x.device)
        self.encoder_fifth_mask = self._get_binary_mask([x.shape[0], self.fc_features], device=x.device)

        self.decoder_first_mask = self._get_binary_mask([x.shape[0], self.fc_features], device=x.device)
        self.decoder_second_mask = self._get_binary_mask([x.shape[0], self.dimension_before_fc], device=x.device)
        self.decoder_third_mask = self._get_binary_mask([x.shape[0], 2*self.base_number_chnanels, 2*self.reshape_channels, 2*self.reshape_channels], device=x.device)
        self.decoder_fourth_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 4*self.reshape_channels, 4*self.reshape_channels], device=x.device)
        self.decoder_fifth_mask = self._get_binary_mask([x.shape[0], self.base_number_chnanels, 8*self.reshape_channels, 8*self.reshape_channels], device=x.device)

    def encoding(self, x, random):

        x = self.encoder_first(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_first_mask 

        x = self.encoder_second(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_second_mask 
            
        x = self.encoder_third(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_third_mask 

        x = self.encoder_fourth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_fourth_mask 

        x = self.flatten(x)

        x = self.encoder_fifth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.encoder_fifth_mask 
            
        x = self.encoder_final(x)

        return x

    def decoding(self, x, random):

        x = self.decoder_first(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_first_mask 

        x = self.decoder_second(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_second_mask 

        x = self.unflatten(x)

        x = self.decoder_third(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_third_mask 

        x = self.decoder_fourth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_fourth_mask 

        x = self.decoder_fifth(x)
        if random:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=True)
        else:
            x = x * self.decoder_fifth_mask 
        
        x = self.decoder_final(x)

        return x

    def forward(self, x, random=True):

        mu = self.encoding(x, random)
        xhat = self.decoding(mu, random)

        return {"xhat":xhat, "mu": mu}

##############################################################################################################################################################


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for ease of use, get the different configs
        self.model_config, self.data_config, self.training_config = config["model"], config["dataset"], config["training"]

        # some definitions
        self.type =  self.model_config["type"]
        self.nbr_input_channels = 1 
        self.pixel_height = self.data_config["img_size"]
        self.pixel_width = self.data_config["img_size"]

        # define the number of classes
        if "sviro" in config["dataset"]["name"].lower():
            if "seats" in config["dataset"]["split"].lower():
                self.nbr_classes = 64
            else:
                self.nbr_classes = 8
        elif config["dataset"]["name"].lower() == "gtsrb":
            self.nbr_classes = 10
        elif config["dataset"]["factor"].lower() == "mnist":
            self.nbr_classes = 10
        elif config["dataset"]["factor"].lower() == "fashion":
            self.nbr_classes = 10
        elif config["dataset"]["factor"].lower() == "svhn":
            self.nbr_classes = 10
            
        # whether to use dropout
        # and width of all layers
        self.dropout_rate = self.model_config["dropout"]  
        self.layer_width = int(self.model_config["width"] * (1+self.dropout_rate))

        # define the activation function to use
        self.activation = get_activation_function(self.model_config)      

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # define the model
        self.classifier = nn.Sequential()
        self.classifier.add_module("fc_1", nn.Linear(in_features=self.pixel_height*self.pixel_width*self.nbr_input_channels, out_features=self.layer_width, bias=True))
        self.classifier.add_module("activation_1", self.activation)
        if self.dropout_rate != 0:  self.classifier.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("fc_2", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.classifier.add_module("activation_2", self.activation)
        if self.dropout_rate != 0:  self.classifier.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("fc_3", nn.Linear(in_features=self.layer_width, out_features=self.layer_width, bias=True))
        self.classifier.add_module("activation_3", self.activation)
        if self.dropout_rate != 0:  self.classifier.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("fc_4", nn.Linear(in_features=self.layer_width, out_features=self.nbr_classes, bias=True))

    def _enable_dropout(self):
        if self.dropout_rate == 0:
            raise ValueError("Enabling dropout does not make any sense.")
        for each_module in self.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def print_model(self):

        print("=" * 57)
        print("The autoencoder is defined as: ")
        print("=" * 57)
        print(self)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)


    def loss(self, prediction, target):
        return self.criterion(prediction, target)

    def forward(self, x):

        # do the reshaping outside the model definition
        # otherwise the jacobian for the reshape layers does not work
        x = x.reshape(-1, self.pixel_height*self.pixel_width*self.nbr_input_channels)
        x = self.classifier(x)

        return {"prediction":x}

##############################################################################################################################################################

class Conv(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for ease of use, get the different configs
        self.model_config, self.data_config, self.training_config = config["model"], config["dataset"], config["training"]

        # define the number of classes
        if "sviro" in config["dataset"]["name"].lower():
            if "seats" in config["dataset"]["split"].lower():
                self.nbr_classes = 64
            else:
                self.nbr_classes = 8
        elif config["dataset"]["name"].lower() == "gtsrb":
            self.nbr_classes = 10
        elif config["dataset"]["name"].lower() == "mnist":
            self.nbr_classes = 10
        elif config["dataset"]["name"].lower() == "fashion":
            self.nbr_classes = 10
        elif config["dataset"]["name"].lower() == "svhn":
            self.nbr_classes = 10

        # some definitions
        self.type =  self.model_config["type"]
        self.dropout_rate = self.model_config["dropout"]
        self.nbr_input_channels = 1 
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels * self.img_size_scale**2


        # define the activation function to use
        self.activation = get_activation_function(self.model_config)      

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # define the model
        self.classifier = nn.Sequential()
        self.classifier.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.classifier.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.classifier.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.classifier.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.classifier.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.classifier.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("flatten", nn.Flatten())
        self.classifier.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.classifier.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.classifier.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.classifier.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.nbr_classes, bias=True))

    def _enable_dropout(self):
        if self.dropout_rate == 0:
            raise ValueError("Enabling dropout does not make any sense.")
        for each_module in self.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def print_model(self):

        print("=" * 57)
        print("The autoencoder is defined as: ")
        print("=" * 57)
        print(self)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)


    def loss(self, prediction, target):
        return self.criterion(prediction, target)

    def forward(self, x):

        # do the reshaping outside the model definition
        # otherwise the jacobian for the reshape layers does not work
        x = self.classifier(x)

        return {"prediction":x}

##############################################################################################################################################################

class ExtractorAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        # get the feature extractor and some related parameters
        self.extractor, self.dimension_before_fc, self.in_channels, self.channels_before_fc = self._get_extractor()

        # get the dropout rate
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels

        # for each of the extracted vgg blocks
        # we do not want to update the weights
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv_1", nn.Conv2d(in_channels=self.in_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=2, stride=2))
        self.encoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True))

        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc * self.img_size_scale**2, bias=True))
        self.decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))

        # interpolate if images to small, i.e. not 224x224 as the images used for pre-training
        self.transform = torch.nn.functional.interpolate

    def _get_extractor(self):

        # for ease of use
        which_model = self.model_config["extractor"]
        which_layer = self.model_config["layer"]

        # get a fixed number of blocks from the pre-trained network
        # make sure it is in eval mode
        if which_model == "vgg11":
            setup = {
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 512, "channels_before_fc":256},
            }
            blocks = list(torchvision.models.vgg11(pretrained=True).eval().features.children())[0:which_layer]

        elif which_model == "resnet50":
            setup = {
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 1024, "channels_before_fc":256},
            }
            blocks = list(torchvision.models.resnet50(pretrained=True).eval().children())[0:which_layer]
        
        elif which_model == "densenet121":
            setup = {
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 1024, "channels_before_fc":256},
            }
            blocks = list(torchvision.models.densenet121(pretrained=True).eval().features.children())[0:which_layer]

        # make the blocks iterable for inference
        extractor = torch.nn.Sequential(*blocks)

        # get the number of input channels for the layers after the extractor
        in_channels = setup[which_layer]["in_channels"]

        # get the number of dimensions as input to the fc
        dimension_before_fc = setup[which_layer]["dimension_before_fc"] 

        # number of channels before fc
        channels_before_fc = setup[which_layer]["channels_before_fc"] 

        return extractor, dimension_before_fc, in_channels, channels_before_fc

    def encode(self, x):

        # if the input does not have 3 channels, i.e. grayscale
        # repeat the image along channel dimension
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        # resize input
        if x.shape[2] != 224:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)

        # extract features by pre-trained network
        x = self.extractor(x)
        
        # encode the extracted features
        x = self.encoder(x)

        return x

    def forward(self, x):

        mu = self.encode(x)
        z = self.decode(mu)

        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class MultiChannelAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        # get the dropout rate
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.channels_before_fc = math.ceil(64 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels * self.img_size_scale**2


        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True))


        # decoder
        self.synthetic_decoder = nn.Sequential()
        self.synthetic_decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.synthetic_decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc, bias=True))
        self.synthetic_decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.synthetic_decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.synthetic_decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.synthetic_decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.synthetic_decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))

        # decoder
        self.real_decoder = nn.Sequential()
        self.real_decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.real_decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc, bias=True))
        self.real_decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.real_decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.real_decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.real_decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.real_decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))


    def decode(self, mu, input_type):

        # decode the extracted features
        if input_type =="synth":
            z = self.synthetic_decoder(mu)
        elif input_type == "real":
            z = self.real_decoder(mu)
        else:
            raise ValueError("Needs to be synth or real.")

        return z

    def forward(self, x, input_type):

        mu = self.encode(x)
        z = self.decode(mu, input_type)
        
        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class MultiChannelExtractorAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        # get the feature extractor and some related parameters
        self.extractor, self.dimension_before_fc, self.in_channels, self.channels_before_fc = self._get_extractor()

        # get the dropout rate
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.channels_before_fc = math.ceil(64 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels

        # for each of the extracted vgg blocks
        # we do not want to update the weights
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv_1", nn.Conv2d(in_channels=self.in_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=2, stride=2))
        self.encoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True))


        # decoder
        self.synthetic_decoder = nn.Sequential()
        self.synthetic_decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.synthetic_decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc * self.img_size_scale**2, bias=True))
        self.synthetic_decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.synthetic_decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.synthetic_decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.synthetic_decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.synthetic_decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.synthetic_decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.synthetic_decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))

        # decoder
        self.real_decoder = nn.Sequential()
        self.real_decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.real_decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc * self.img_size_scale**2, bias=True))
        self.real_decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.real_decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.real_decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.real_decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.real_decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.real_decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.real_decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))

        # interpolate if images to small, i.e. not 224x224 as the images used for pre-training
        self.transform = torch.nn.functional.interpolate

    def _get_extractor(self):

        # for ease of use
        which_model = self.model_config["extractor"]
        which_layer = self.model_config["layer"]

        # get a fixed number of blocks from the pre-trained network
        # make sure it is in eval mode
        if which_model == "vgg11":
            setup = {
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 512, "channels_before_fc":256},
            }
            blocks = list(torchvision.models.vgg11(pretrained=True).eval().features.children())[0:which_layer]

        elif which_model == "resnet50":
            setup = {
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 1024, "channels_before_fc":256},
            }
            blocks = list(torchvision.models.resnet50(pretrained=True).eval().children())[0:which_layer]
        
        elif which_model == "densenet121":
            setup = {
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 1024, "channels_before_fc":256},
            }
            blocks = list(torchvision.models.densenet121(pretrained=True).eval().features.children())[0:which_layer]

        # make the blocks iterable for inference
        extractor = torch.nn.Sequential(*blocks)

        # get the number of input channels for the layers after the extractor
        in_channels = setup[which_layer]["in_channels"]

        # get the number of dimensions as input to the fc
        dimension_before_fc = setup[which_layer]["dimension_before_fc"] 

        # number of channels before fc
        channels_before_fc = setup[which_layer]["channels_before_fc"] 

        return extractor, dimension_before_fc, in_channels, channels_before_fc

    def encode(self, x):

        # if the input does not have 3 channels, i.e. grayscale
        # repeat the image along channel dimension
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        # resize input
        if x.shape[2] != 224:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)

        # extract features by pre-trained network
        x = self.extractor(x)
        
        # encode the extracted features
        x = self.encoder(x)

        return x

    def decode(self, mu, input_type):

        # decode the extracted features
        if input_type =="synth":
            z = self.synthetic_decoder(mu)
        elif input_type == "real":
            z = self.real_decoder(mu)
        else:
            raise ValueError("Needs to be synth or real.")

        return z

    def forward(self, x, input_type):

        mu = self.encode(x)
        z = self.decode(mu, input_type)
        
        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class ConvVAE(BaseAE):
    def __init__(self, config):
        # call the init function of the parent class
        super().__init__(config)

        # get the dropout rate
        self.dropout_rate = self.model_config["dropout"] 

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.base_number_chnanels = math.ceil(32 * (1+self.dropout_rate))
        self.channels_before_fc = math.ceil(64 * (1+self.dropout_rate))
        self.fc_features = math.ceil(256 * (1+self.dropout_rate))

        # increase the number of channels and fc neurons based on the dropout rate
        # we need to round up because pytorch seems to round up as well
        self.reshape_channels = 4
        self.img_size_scale = int(self.data_config["img_size"] / 64)
        self.dimension_before_fc = 2 * self.base_number_chnanels * self.reshape_channels * self.reshape_channels * self.img_size_scale**2

        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv_1", nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_2", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_3", nn.Conv2d(in_channels=self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("conv_4", nn.Conv2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.encoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("fc_1", nn.Linear(in_features=self.dimension_before_fc, out_features=self.fc_features, bias=True))
        self.encoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.encoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))

        self.encoder_fc21 = nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True)
        self.encoder_fc22 = nn.Linear(in_features=self.fc_features, out_features=self.latent_dim, bias=True)

        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module("fc_1", nn.Linear(in_features=self.latent_dim, out_features=self.fc_features, bias=True))
        self.decoder.add_module("activation_1", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_1", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("fc_2", nn.Linear(in_features=self.fc_features, out_features=self.dimension_before_fc, bias=True))
        self.decoder.add_module("activation_2", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_2", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("unflatten", nn.Unflatten(1, (2 * self.base_number_chnanels, self.reshape_channels * self.img_size_scale, self.reshape_channels * self.img_size_scale)))
        self.decoder.add_module("conv_trans_1", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=2*self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_3", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_3", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_2", nn.ConvTranspose2d(in_channels=2*self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_4", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_4", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_3", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.base_number_chnanels, kernel_size=4, padding=1, stride=2))
        self.decoder.add_module("activation_5", self.activation)
        if self.dropout_rate != 0: self.decoder.add_module("dropout_5", nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module("conv_trans_4", nn.ConvTranspose2d(in_channels=self.base_number_chnanels, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2))


    def kl_divergence_loss(self, mu, logvar):

        # KL divergence between gaussian prior and sampled vector
        # we need to sum over the latent dimension
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)

        # take the mean over the batch dimension
        kl_loss = kl_loss.mean(dim=0)

        return kl_loss

    def reparametrize(self, mu, logvar):
        
        # reparametrization rick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def encode(self, x):

        x = self.encoder(x)
        mu = self.encoder_fc21(x)
        logvar = self.encoder_fc21(x)

        return mu, logvar

    def decode(self, z):

        z = self.decoder(z)

        return z

    def sample(self, mu, logvar):

        # if we want to reparametrize and we are in training mode
        if self.training:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu

        return z

    def forward(self, x):

        # encode the input image
        mu, logvar = self.encode(x)

        # sample the latent vector
        z = self.sample(mu, logvar)

        # decode the sampled latent vector
        xhat = self.decode(z)
        
        return {"xhat": xhat, "mu": mu, "logvar": logvar, "z":z}

##############################################################################################################################################################
