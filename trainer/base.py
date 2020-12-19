from pathlib import Path
import os
import pickle
import time

import torch

from domain.metadata import ModelFileMetadata
from logger import logger
from properties import APPLICATION_PROPERTIES
from utils import Utils

time.time()


class TrainerBase(object):

    def __init__(self, model, model_file_metadata: ModelFileMetadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env='script', _logger=logger, is_saved=True):
        self.model = model
        self.model_file_metadata = model_file_metadata
        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader else test_loader
        self.test_loader = test_loader
        self.hyperparameters = hyperparameters
        self.logger = _logger

        # Set environments
        self.tqdm = None
        self.is_plot_showed = False
        self.tqdm_disable = False

        self.set_tqdm_env(tqdm_env=tqdm_env)

        if is_saved:
            self.create_model_directory()

        # Set model configuration
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.hyperparameters.device)
            logger.info(f"Model set to '{self.hyperparameters.device}'")

        self.best_model = dict()

    def set_tqdm_env(self, tqdm_env):
        tqdm_env_dict = Utils.get_tqdm_env_dict(tqdm_env=tqdm_env)

        self.tqdm = tqdm_env_dict["tqdm"]
        self.is_plot_showed = tqdm_env_dict["tqdm_disable"]
        self.tqdm_disable = tqdm_env_dict["is_plot_showed"]

    def create_model_directory(self):
        Path(self.model_file_metadata.model_dir_path).mkdir(parents=True, exist_ok=True)

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.model_file_metadata.get_save_model_file_path(epoch=epoch))

    def save_record(self, record, epoch):
        with open(self.model_file_metadata.get_save_record_file_path(epoch=epoch), "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, epoch):
        model_file_path = self.model_file_metadata.get_save_model_file_path(epoch=epoch)
        self._load_model_with_path(model_file_path=model_file_path)

    def get_record(self, epoch):
        record = None

        with open(self.model_file_metadata.get_save_record_file_path(epoch=epoch), "rb") as f:
            pickle.load(f)

        return record

    def load_best_model(self):
        best_model_file_path = self.model_file_metadata.get_best_model_file_path()
        self._load_model_with_path(model_file_path=best_model_file_path)

    def get_entire_record_file(self):
        entire_record_file = None
        entire_record_file_path = self.model_file_metadata.get_entire_record_file_path()

        if os.path.isfile(entire_record_file_path):
            with open(entire_record_file_path, "rb") as f:
                entire_record_file = pickle.load(f)

            logger.info(f"Succeed to get entire record file")
        else:
            logger.error(f"Failed to get entire record file, file not exist")

        return entire_record_file

    def _load_model_with_path(self, model_file_path):
        if os.path.isfile(model_file_path):
            self.model.load_state_dict(torch.load(model_file_path, map_location=APPLICATION_PROPERTIES.DEVICE_CPU))
            self.model.to(self.hyperparameters.device)

            logger.info(f"Succeed to load best model, device: '{self.hyperparameters.device}'")
        else:
            logger.error(f"Failed to load best model, file not exist")

    def train(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class OCCTrainerBase(TrainerBase):

    def __init__(self, model, model_file_metadata, hyperparameters, tqdm_env='script'):
        super(OCCTrainerBase, self).__init__(model, model_file_metadata, None, None, None, hyperparameters, tqdm_env)

    def save_model(self, epoch):
        with open(self.model_file_metadata.get_save_model_file_path(epoch=epoch), "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, epoch):
        model = None

        with open(self.model_file_metadata.get_save_model_file_path(epoch=epoch), "rb") as f:
            pickle.load(f)

        return model

    def load_best_model(self):
        best_model_file_path = self.model_file_metadata.get_best_model_file_path()

        if os.path.isfile(best_model_file_path):
            with open(best_model_file_path, "rb") as f:
                self.model = pickle.load(f)

            logger.info(f"Succeed to load best model, device: '{self.hyperparameters.device}'")
        else:
            logger.error(f"Failed to load best model, file not exist")


class HybridTrainerBase(TrainerBase):

    def __init__(self, encoder_model, encoder_model_file_metadata, occ_trainer, train_loader, val_loader, test_loader, hyperparameters, tqdm_env='script'):
        super(HybridTrainerBase, self).__init__(encoder_model, encoder_model_file_metadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env)
        self.occ_trainer: OCCTrainerBase = occ_trainer

    def load_encoder_model(self, model_file_path):
        if os.path.isfile(model_file_path):
            self.model.load_state_dict(torch.load(model_file_path, map_location=APPLICATION_PROPERTIES.DEVICE_CPU))
            self.model.to(self.hyperparameters.device)

            logger.debug(f"Succeed to load model, {os.path.basename(model_file_path)}, device: '{self.hyperparameters.device}'")
        else:
            logger.error(f"Failed to load model, file not exist")
