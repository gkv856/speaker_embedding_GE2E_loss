import torch.nn.functional as F
import torch
import os
import time
import numpy as np

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

        # creating the folder to save checkpoints
        chk_path = os.path.join(hp.general.project_root, hp.m_avc.tpm.checkpoint_dir)
        os.makedirs(chk_path, exist_ok=True)

        # loss list
        self.train_losses = []

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

    # ================================================================================================================#

    def start_training(self):
        # Print logs in specified order
        keys = ['Reconst', 'Reconst_Resi', 'Content_loss']

        # Start training.
        print('Start training...')

        # setting the model to training mode
        self.auto_vc_net = self.auto_vc_net.train()

        training_st = time.time()
        flushed = True
        for e in range(self.hp.m_avc.tpm.num_iters):

            if flushed:
                print(f"Epoch:[{e + self.hp.m_avc.tpm.log_step}/{self.hp.m_avc.tpm.num_iters}] ", end="")
                flushed = False

            # creating empty loss list
            bl_reconst_mel_spect = []
            bl_recon_mel_spect_inc_residual = []
            bl_spkr_content = []

            # there are 24 speakers therefore depending upon the batch size, this loop will run x number of times
            # starting with batch_size = 1 therefore it will run 24 times.
            # each time it will pick random utterance for each 24 speakers

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
            # emb = torch.tensor(emb).float()

            utter_specs = utter_specs.to(self.hp.general.device)
            emb = emb.to(self.hp.general.device)

            # Calculating total reconst loss
            # here we are using the original mel-spects
            ypred_mel_spects, ypred_mel_spects_final, ypred_spkr_content = self.auto_vc_net(utter_specs, emb, emb)
            loss_reconst_mel_spect = F.mse_loss(utter_specs, ypred_mel_spects)
            loss_recon_mel_spect_inc_residual = F.mse_loss(utter_specs, ypred_mel_spects_final)

            # calculating loss for only speaker's content not the style/emb
            # here we are using the predicted mel-spects
            # since we are calculating 'loss_recon_mel_spect_inc_residual'
            # it should not matter if we use 'ypred_mel_spects' or 'ypred_mel_spects_final'
            ypred_spkr_content_only = self.auto_vc_net(ypred_mel_spects_final, emb, None)
            loss_spkr_content = F.l1_loss(ypred_spkr_content_only, ypred_spkr_content)

            # Backward and optimize.
            loss_total = loss_reconst_mel_spect + \
                         loss_recon_mel_spect_inc_residual + \
                         self.hp.m_avc.tpm.lambda_cd * loss_spkr_content

            self.reset_grad()
            loss_total.backward()
            self.optimizer.step()

            # collecting loss per batch
            bl_reconst_mel_spect.append(loss_reconst_mel_spect.item())
            bl_recon_mel_spect_inc_residual.append(loss_recon_mel_spect_inc_residual.item())
            bl_spkr_content.append(loss_spkr_content.item())
            # print(loss_reconst_mel_spect.item(), loss_recon_mel_spect_inc_residual.item(), loss_spkr_content.item())

            # Logging.
            loss = {}
            loss['Reconst'] = np.mean(bl_reconst_mel_spect)
            loss['Reconst_Resi'] = np.mean(bl_recon_mel_spect_inc_residual)
            loss['Content_loss'] = np.mean(bl_spkr_content)
            curr_batch_loss = (loss['Reconst'], loss['Reconst_Resi'], loss['Content_loss'])
            self.train_losses.append(curr_batch_loss)

            if (e + 1) % self.hp.m_avc.tpm.dot_print == 0:
                print(".", end="")

            if (e + 1) % self.hp.m_avc.tpm.log_step == 0:

                # calculating epoch running time
                if e == self.hp.m_avc.tpm.log_step - 1:
                    epoch_st = training_st

                epoch_et = time.time()
                hours, rem = divmod(epoch_et - epoch_st, 3600)
                minutes, seconds = divmod(rem, 60)
                time_msg = ", {:0>2}:{:0>2}:{:0.0f}".format(int(hours), int(minutes), seconds)

                msg = ""
                # collecting the loss message
                for tag in keys:
                    msg += " {}: {:.4f}".format(tag, loss[tag])

                msg = msg + time_msg
                print(msg, end="\n")

                flushed = True
                epoch_st = epoch_et

            if (e + 1) % self.hp.m_avc.tpm.reduce_lr_interval == 0:
                print(f"Reducing learning rate from {self.lr} to {self.lr / 2}")
                self.lr = self.lr / 2
                self.optimizer.param_groups[0]['lr'] = self.lr

            if (e + 1) % self.hp.m_avc.tpm.checkpoint_interval == 0:
                self.__save_model(e, verbose=True)

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
