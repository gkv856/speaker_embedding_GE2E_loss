import torch
import os
import random
import time
import numpy as np

try:
    from embedding_model_GE2E.s1_dataset_loader import get_train_test_data_loader
    from embedding_model_GE2E.s2_model_GE2E_loss_speach_embed import ModelGE2ELossSpeachEmbed
    from embedding_model_GE2E.s3_loss_function_GE2E import GE2ELoss
except:
    from AVC.s2_generalized_end2end_loss_GE2E.s1_dataset_loader import get_train_test_data_loader
    from AVC.s2_generalized_end2end_loss_GE2E.s2_model_GE2E_loss_speach_embed import ModelGE2ELossSpeachEmbed
    from AVC.s2_generalized_end2end_loss_GE2E.s3_loss_function_GE2E import GE2ELoss


class TrainEmbedModel:

    def __init__(self, hp):
        # creating a fresh model
        self.model = ModelGE2ELossSpeachEmbed(hp).to(hp.general.device)

        # loading weights if pre-trained is true
        if hp.m_ge2e.restore_existing_model:
            model_path = os.path.join(hp.general.project_root, hp.m_ge2e.model_path)

            # load weights as dictionary
            weight_dict = torch.load(model_path, map_location=hp.general.device)
            self.model.load_state_dict(weight_dict)
            print(f"Pre-trained model loaded {model_path}")

        # creating GE2E loss object
        self.ge2e_loss = GE2ELoss(hp)
        # Both net and loss have trainable parameters
        lst_train_params_for = [
            {'params': self.model.parameters()},
            {'params': self.ge2e_loss.parameters()}
        ]

        # creating the SGD optimizer
        self.lr = hp.m_ge2e.lr
        self.optimizer = torch.optim.SGD(lst_train_params_for, lr=self.lr)

        # creating train and test loaders
        self.train_loader, self.test_loader = get_train_test_data_loader(hp)

        # creating the folder to save checkpoints
        chk_path = os.path.join(hp.general.project_root, hp.m_ge2e.checkpoint_dir)
        os.makedirs(chk_path, exist_ok=True)

        # creating global lists for
        self.train_losses = []
        self.test_losses = []

        # total utterrances
        self.total_utterances = hp.m_ge2e.training_N * hp.m_ge2e.training_M

        # storing hp
        self.hp = hp

    def __get_batched_test_loss(self, hp):
        """
        calculates the batched loss using the test data
        :param hp:
        :return:
        """

        # switching model to test mode
        self.model.eval()

        # calc total utterances = speakers x utters
        test_total_utterances = hp.m_ge2e.test_N * hp.m_ge2e.test_M

        test_batch_loss = []
        for test_mel_db_batch in self.test_loader:
            # sending data to GPU/TPU for calculation
            test_mel_db_batch = test_mel_db_batch.to(hp.general.device)

            # mel is returned as 4x5x160x40 (batch x num_speaker x utterlenxn_mel)
            # and we will reshape it to 20x160x40
            new_shape = (test_total_utterances, test_mel_db_batch.shape[2], test_mel_db_batch.shape[3])
            test_mel_db_batch = torch.reshape(test_mel_db_batch, new_shape)

            perm = random.sample(range(0, test_total_utterances), test_total_utterances)
            unperm = list(perm)

            # saving the unpermutated status of the utterances
            # this will be used to fetch correct utterance per person
            for i, j in enumerate(perm):
                unperm[j] = i

            test_mel_db_batch = test_mel_db_batch[perm]

            # passing the test data through the model to get the speaker embeddings
            embeddings = self.model(test_mel_db_batch)
            embeddings = embeddings[unperm]

            # changing the shape back num_speakers x utter_per_speaker x embedding_vector
            embeddings = torch.reshape(embeddings, (hp.m_ge2e.test_N, hp.m_ge2e.test_M, embeddings.shape[1]))

            # get loss, call backward, step optimizer
            # shape should be (Speaker, Utterances, embedding)
            test_loss = self.ge2e_loss(embeddings)
            test_batch_loss.append(test_loss.to("cpu").detach().numpy())

        # switching model to training mode
        self.model.train()

        mean_curr_test_batch_loss = np.mean(test_batch_loss)
        return mean_curr_test_batch_loss

    def __save_model(self, hp, e, loss, name="ckpt_epoch", verbose=True):
        """
        this method saves a model checkpoint during the training
        :param hp: hyper parameters
        :param e: number of eopchs
        :param loss: current training loss
        :param name:
        :param verbose: to print the info or not
        :return:
        """
        # switching the model back to test mode
        self.model.eval().cpu()

        # creating chk pt name
        ckpt_model_filename = f"{name}_{e + 1}_L_{loss:.4f}.pth"
        ckpt_model_path = os.path.join(hp.general.project_root, hp.m_ge2e.checkpoint_dir, ckpt_model_filename)

        # saving the file
        torch.save(self.model.state_dict(), ckpt_model_path)
        if verbose:
            print(f"Model saved as '{ckpt_model_filename}'")

        # switching the model back to train model
        self.model.to(hp.general.device).train()

    def train_model(self, lr_reduce=2000, epoch_print=100, dot_print=10):
        """
        this method trains the embedding model
        :param hp: hyper parameters
        :param lr_reduce: reduce lr to half after these many epochs
        :param epoch_print: print the loss info after these many epochs
        :param dot_print: print a dot during training to show progress
        :return: model, losses
        """

        hp = self.hp
        # setting model to train mode
        self.model.train()

        # setting up the variables
        best_test_batch_loss = None

        # setting the timer
        training_st = time.time()

        flushed = True
        for e in range(hp.m_ge2e.training_epochs):

            if flushed:
                print(f"Epoch:[{e + epoch_print}/{hp.m_ge2e.training_epochs}] ", end="")
                flushed = False

            # flushing the batched training loss
            batch_train_loss = []
            # depending upon the 'hp.training_N' this loop will run n number of times. n = total speakers/hp.training_N
            for mel_db_batch in self.train_loader:

                # sending data to GPU/TPU for calculation
                mel_db_batch = mel_db_batch.to(hp.general.device)

                # mel is returned as 4x5x160x40 (batch x num_speaker x utterlen x n_mel)
                # and we will reshape it to 20x160x40
                new_shape = (self.total_utterances, mel_db_batch.shape[2], mel_db_batch.shape[3])
                mel_db_batch = torch.reshape(mel_db_batch, new_shape)

                perm = random.sample(range(0, self.total_utterances), self.total_utterances)
                unperm = list(perm)

                # saving the unpermutated status of the utterances,
                # this will be used to fetch correct utterance per person
                for i, j in enumerate(perm):
                    unperm[j] = i

                mel_db_batch = mel_db_batch[perm]

                # passing the data through the model to get the embeddings
                embeddings = self.model(mel_db_batch)
                embeddings = embeddings[unperm]

                # changing the shape back num_speakers x utter_per_speaker x embedding_vector
                embeddings = torch.reshape(embeddings, (hp.m_ge2e.training_N, hp.m_ge2e.training_M, embeddings.size(1)))

                # calculating the loss
                # embeddings shape should be = (Speaker, Utterances, embedding)
                loss = self.ge2e_loss(embeddings)

                # back prop and normalization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                torch.nn.utils.clip_grad_norm_(self.ge2e_loss.parameters(), 1.0)
                self.optimizer.step()

                batch_train_loss.append(loss.to("cpu").detach().numpy())

            #############################################################
            #                                                           #
            #               printing epoch info                         #
            #                                                           #
            #############################################################

            # calculating mean of the batch loss
            mean_batch_train_loss = np.mean(batch_train_loss)

            # list of all the losses for each epoch
            self.train_losses.append(mean_batch_train_loss)

            if (e + 1) % dot_print == 0:
                print(".", end="")

            if (e + 1) % epoch_print == 0:

                # test model loss against the test data
                mean_curr_test_batch_loss = self.__get_batched_test_loss(hp)
                self.test_losses.append(mean_curr_test_batch_loss)

                # calculating epoch running time
                if e == epoch_print - 1:
                    epoch_st = training_st

                epoch_et = time.time()
                hours, rem = divmod(epoch_et - epoch_st, 3600)
                minutes, seconds = divmod(rem, 60)
                time_msg = "{:0>2}:{:0>2}:{:0.0f}".format(int(hours), int(minutes), seconds)

                msg = " Train_Loss:{0:.4f}\t".format(mean_batch_train_loss)
                msg = msg + " Test_Loss:{0:.4f}\t".format(mean_curr_test_batch_loss)
                msg = msg + time_msg

                print(msg, end="\n")

                # saving the best test loss model
                if hp.m_ge2e.save_best_weights and mean_curr_test_batch_loss < hp.m_ge2e.min_test_loss:
                    # is this is first epoch then simple curr test loss is the best test loss
                    if best_test_batch_loss is None:
                        best_test_batch_loss = mean_curr_test_batch_loss

                    # else check if curr loss is less than best loss, if yes then replace
                    elif best_test_batch_loss > mean_curr_test_batch_loss:
                        best_test_batch_loss = mean_curr_test_batch_loss

                    # if curr loss = best loss then save the model
                    if best_test_batch_loss == mean_curr_test_batch_loss:
                        self.__save_model(hp, e, best_test_batch_loss, name="m_best", verbose=False)

                flushed = True
                epoch_st = epoch_et

            # reducing the learning rate by half
            if (e + 1) % lr_reduce == 0:
                print(f"Reducing learning rate from {self.lr} to {self.lr / 2}")
                self.lr = self.lr / 2
                self.optimizer.param_groups[0]['lr'] = self.lr

            if hp.m_ge2e.checkpoint_dir is not None and (e + 1) % hp.m_ge2e.checkpoint_interval == 0:
                self.__save_model(hp, e, mean_batch_train_loss, verbose=True)

        # save final model
        self.__save_model(hp, e, mean_batch_train_loss, name="final_epoch", verbose=True)

        return self.model, self.train_losses, self.test_losses


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass
    #
    # from strings.constants import hp
    #
    # hp.m_ge2e.training_epochs = 100
    # hp.m_ge2e.checkpoint_interval = 10
    #
    # # creating training object
    # train_emb_model_obj = TrainEmbedModel(hp)
    #
    # # training the model
    # model, train_loss, test_loss = train_emb_model_obj.train_model(lr_reduce=20, epoch_print=10, dot_print=1)
    # print(1)