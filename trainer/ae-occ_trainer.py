from copy import deepcopy
import pickle
import time

import torch

from trainer.base import HybridTrainerBase

time.time()


class HybridAETrainer(HybridTrainerBase):

    def __init__(self, encoder_model, encoder_model_file_metadata, occ_trainer, train_loader, val_loader, test_loader, hyperparameters, tqdm_env='script'):
        super(HybridAETrainer, self).__init__(encoder_model, encoder_model_file_metadata, occ_trainer, train_loader, val_loader, test_loader, hyperparameters, tqdm_env)

    def train(self):
        self.model.eval()

        train_result_dict_list = list()
        val_result_dict_list = list()

        best_auc = 0
        best_auc_epoch = 0

        # Set hyperparameters
        n_epoch = self.hyperparameters.n_epoch
        device = self.hyperparameters.device

        for epoch in self.tqdm.tqdm(range(n_epoch)):

            current_model_file_path = self.model_file_metadata.get_save_model_file_path(epoch=epoch)
            self.load_encoder_model(model_file_path=current_model_file_path)

            for i, (data_batch, label_batch, abnormal_label_batch) in enumerate(self.train_loader):
                data_batch = data_batch.to(device)
                pred_data_batch, latent_data_batch = self.model(data_batch)

                latent_data_batch = latent_data_batch.detach().numpy()

                with torch.no_grad():
                    self.occ_trainer.train(x_train=latent_data_batch)

            train_result_dict = dict()
            val_result_dict = self.validate()

            if best_auc < val_result_dict['auc']:
                best_auc = val_result_dict['auc']
                best_auc_epoch = epoch
                self.best_model = deepcopy(self.occ_trainer.model)

            # Print
            print(
                f"[Epoch {epoch}] "
                f"Val - AUC : {round(val_result_dict['auc'], 7)} | "
                f"Best AUC : {round(best_auc, 7)} (epoch : {best_auc_epoch})"
            )

            # Save Model & Record dictW
            record_dict = dict(
                train_result_dict=train_result_dict,
                val_result_dict=val_result_dict
            )

            self.occ_trainer.save_model(epoch=epoch)
            self.occ_trainer.save_record(record=record_dict, epoch=epoch)

            train_result_dict_list.append(train_result_dict)
            val_result_dict_list.append(val_result_dict)

        # Save last result
        entire_record_dict = dict(
            train_result_dict_list=train_result_dict_list,
            val_result_dict_list=val_result_dict_list
        )

        # Save best model
        with open(self.model_file_metadata.get_best_model_file_path(), "wb") as f:
            pickle.dump(self.best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save entire_record_dict
        with open(self.model_file_metadata.get_entire_record_file_path(), "wb") as f:
            pickle.dump(entire_record_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Load best model
        self.occ_trainer.load_best_model()

        return entire_record_dict

    def validate(self):
        self.model.eval()

        n_batch = 0

        latent_data_list = list()
        abnormal_label_list = list()

        # Set hyperparameters
        device = self.hyperparameters.device

        for i, (data_batch, label_batch, abnormal_label_batch) in enumerate(self.val_loader):
            data_batch = data_batch.to(device)
            abnormal_label_batch = abnormal_label_batch.to(device)

            with torch.no_grad():
                pred_data_batch, latent_data_batch = self.model(data_batch)

            latent_data_list.append(latent_data_batch)
            abnormal_label_list.append(abnormal_label_batch)

            # total_loss += loss
            n_batch += 1

        latent_data_list = torch.cat(latent_data_list).cpu().numpy()
        abnormal_label_list = torch.cat(abnormal_label_list).cpu().numpy()

        occ_result_dict = self.occ_trainer.validate(x_test=latent_data_list, y_test=abnormal_label_list)

        return dict(auc=occ_result_dict["auc"], latent_data_list=latent_data_list)
