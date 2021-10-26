import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from s3_auto_voice_cloner.s2_auto_vc_dataloader import get_auto_vc_data_loader
    from s3_auto_voice_cloner.s4_auto_vc_network import AutoVCNetwork
except:
    from AVC.s3_auto_voice_cloner.s2_auto_vc_dataloader import get_auto_vc_data_loader
    from AVC.s3_auto_voice_cloner.s4_auto_vc_network import AutoVCNetwork


class TrainAutoVCNetwork(object):

    def __init__(self, hp):
        """Initialize configurations."""

        # to be used later
        self.hp = hp

        # create data loader.
        self.data_loader = get_auto_vc_data_loader(hp, batch_size=hp.m_avc.tpm.data_batch_size)

        # learing rate
        self.lr = hp.m_avc.tpm.lr

        # Build the model and tensorboard.
        self.__create_model()

        # setting the model to training mode
        self.auto_vc_net = self.auto_vc_net.train()

        # creating the folder to save checkpoints
        chk_path = os.path.join(hp.general.project_root, hp.m_avc.tpm.checkpoint_dir)
        os.makedirs(chk_path, exist_ok=True)

        # loss list
        self.train_losses = []

        # curr loss
        self.curr_loss = {}

        # curr loss
        self.curr_loss = {
            "l_recon": 0.0,
            "l_recon0": 0.0,
            "l_content": 0.0
        }

        # batched loss list
        # creating empty loss list
        self.bl_l_recon = []
        self.bl_l_recon0 = []
        self.bl_l_content = []

        # for print status
        self.flushed = True
        self.epoch_st = None
        self.epoch_ed = None
        self.training_st = None

    def __save_model(self, e, name="ckpt_epoch", verbose=True):
        """
        this method saves a model checkpoint during the training
        :param hp: hyper parameters
        :param e: number of eopchs
        :param name:
        :param verbose: to print the info or not
        :return:
        """
        # switching the model back to test mode
        self.auto_vc_net.eval().cpu()

        # creating chk pt name
        ckpt_model_filename = f"{name}_{e + 1}.pth"
        ckpt_model_path = os.path.join(self.hp.general.project_root, self.hp.m_avc.tpm.checkpoint_dir,
                                       ckpt_model_filename)

        # saving the file
        torch.save(self.auto_vc_net.state_dict(), ckpt_model_path)
        if verbose:
            print(f"Model saved as '{ckpt_model_filename}'")

        # switching the model back to train model
        self.auto_vc_net.to(self.hp.general.device).train()

    def __create_model(self):
        """
        creates Auto Voice Cloner and the optimizer
        :return:
        """

        # creating an instance of the AutoVC network
        # sending the model to the GPU (if available)
        self.auto_vc_net = AutoVCNetwork(self.hp).to(self.hp.general.device)

        # creating an adam optimizer
        self.optimizer = torch.optim.Adam(self.auto_vc_net.parameters(), self.lr)

    def reset_grad(self):
        """
        Reset the gradient buffers.
        :return:
        """
        self.optimizer.zero_grad()

    def reduce_lr(self):
        """
        this function reduces the learning rate use in the Adam optimizer by half until it reaches to '0.0001'
        after that, no changes in the learning rate.
        :return:
        """
        new_reduced_lr = self.lr / 2
        if new_reduced_lr > 0.0001:
            new_reduced_lr = self.lr / 2
            print(f"Reducing learning rate from '{self.lr}' to '{new_reduced_lr}'")
        else:
            new_reduced_lr = 0.0001
            if self.lr != 0.0001:
                print(f"Fixing learning rate to '{new_reduced_lr}'")

        self.lr = new_reduced_lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def print_training_info(self, e):
        """
        this function prints the training info to the console
        :param e: current epoch count
        :return:
        """

        curr_batch_loss = (self.curr_loss['l_recon'], self.curr_loss['l_recon0'], self.curr_loss['l_content'])
        self.train_losses.append(curr_batch_loss)

        if e % self.hp.m_avc.tpm.dot_print == 0:
            print(".", end="")

        if e % self.hp.m_avc.tpm.log_step == 0:

            # calculating epoch running time
            if e == self.hp.m_avc.tpm.log_step:
                self.epoch_st = self.training_st

            self.epoch_et = time.time()
            hours, rem = divmod(self.epoch_et - self.epoch_st, 3600)
            minutes, seconds = divmod(rem, 60)
            time_msg = ", {:0>2}:{:0>2}:{:0.0f}".format(int(hours), int(minutes), seconds)

            msg = ""
            # collecting the loss message
            for k, v in self.curr_loss.items():
                msg += " {}: {:.4f},".format(k, v)

            msg = msg + time_msg
            print(msg, end="\n")

            self.flushed = True
            self.epoch_st = self.epoch_et

        if e % self.hp.m_avc.tpm.reduce_lr_interval == 0:
            self.reduce_lr()

        if e % self.hp.m_avc.tpm.checkpoint_interval == 0:
            self.__save_model(e, verbose=True)

    def avc_forward_backprop_step(self, utter_specs, emb):
        """
        this function runs the data through the model to get the prediction (forward)
        and then uses backpropagation to update the weights
        :param utter_specs: a speakers utterance
        :param emb: speaker embedding
        :return:
        """

        utter_specs = utter_specs.to(self.hp.general.device)
        emb = emb.to(self.hp.general.device)

        # Calculating total reconst loss
        # here we are using the original mel-spects
        ypred_mel_spects, ypred_mel_spects_final, ypred_spkr_content = self.auto_vc_net(utter_specs, emb, emb)
        # L_recon = F.mse_loss(utter_specs, ypred_mel_spects)
        L_recon0 = F.mse_loss(utter_specs, ypred_mel_spects_final)

        # calculating loss for only speaker's content not the style/emb
        # here we are using the predicted mel-spects
        # since we are calculating 'loss_recon_mel_spect_inc_residual'
        # it should not matter if we use 'ypred_mel_spects' or 'ypred_mel_spects_final'
        ypred_spkr_content_only = self.auto_vc_net(ypred_mel_spects_final, emb, None)
        L_content = F.l1_loss(ypred_spkr_content_only, ypred_spkr_content)

        # Backward and optimize.
        # loss_total = L_recon + L_recon0 + self.hp.m_avc.tpm.lambda_cd * L_content
        loss_total = L_recon0 + self.hp.m_avc.tpm.lambda_cd * L_content

        self.reset_grad()
        loss_total.backward()
        self.optimizer.step()

        return L_recon0.item(), L_content.item()

    # ================================================================================================================#

    def start_training(self, batched=True):

        # Start training.
        if batched:
            print('Started Batched Training...')
        else:
            print('Started Non-Batched Training...')

        # setting the training time
        self.training_st = time.time()

        for e in range(self.hp.m_avc.tpm.num_iters):

            if self.flushed:
                print(f"Epoch:[{e + self.hp.m_avc.tpm.log_step}/{self.hp.m_avc.tpm.num_iters}] ", end="")
                self.flushed = False

            # creating empty loss list
            self.bl_l_recon = []
            self.bl_l_recon0 = []
            self.bl_l_content = []

            if batched:
                # if there are 24 speakers then depending upon the batch size, this loop will run x number of times
                # starting with batch_size = 1 therefore it will run 24 times.
                # each time it will pick random utterance for each 24 speakers

                for utter_specs, emb, spr in self.data_loader:
                    # for utter_specs, emb, spr in self.data_loader:
                    # if len(utter_specs):
                    utter_specs = utter_specs.clone().detach().cpu().requires_grad_(True).float()
                    emb = emb.clone().detach().cpu().requires_grad_(True).float()

                    # call model training step
                    l_recon0, l_content = self.avc_forward_backprop_step(utter_specs, emb)

                    # collecting loss per batch
                    self.bl_l_recon.append(0.0)
                    self.bl_l_recon0.append(l_recon0)
                    self.bl_l_content.append(l_content)

            else:

                # Fetch data.
                try:
                    utter_specs, emb, spr = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    utter_specs, emb, spr = next(data_iter)

                # for utter_specs, emb, spr in self.data_loader:
                # if len(utter_specs):
                utter_specs = utter_specs.clone().detach().cpu().requires_grad_(True).float()
                emb = emb.clone().detach().cpu().requires_grad_(True).float()

                # call model training step
                l_recon0, l_content = self.avc_forward_backprop_step(utter_specs, emb)

                # collecting loss per batch
                self.bl_l_recon.append(0.0)
                self.bl_l_recon0.append(l_recon0)
                self.bl_l_content.append(l_content)

            # Logging.
            self.curr_loss['l_recon'] = np.mean(self.bl_l_recon)
            self.curr_loss['l_recon0'] = np.mean(self.bl_l_recon0)
            self.curr_loss['l_content'] = np.mean(self.bl_l_content)

            self.print_training_info(e + 1)

        # save final model
        self.__save_model(e, name="final", verbose=True)

        return self.auto_vc_net, self.train_losses


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    # from strings.constants import hp

    # hp.m_avc.tpm.lambda_cd = 1
    # hp.m_avc.tpm.num_iters = 10
    # hp.m_avc.tpm.log_step = 1
    # hp.m_avc.tpm.dot_print = 1
    # hp.m_avc.tpm.checkpoint_interval = 2
    # hp.m_avc.tpm.lr = 0.001
    # hp.m_avc.tpm.reduce_lr_interval = 1
    # hp.m_avc.tpm.data_batch_size = 2
    #
    # solver = TrainAutoVCNetwork(hp)
    #
    # # start the training
    # auto_vc_model, lst_loss_tuple = solver.start_training()
    #
    print(1)
