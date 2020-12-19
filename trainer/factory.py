from torch import optim
from torch.nn import functional as F
import torch

from domain.hyperparameters import Hyperparameters
from domain.metadata import ModelFileMetadata
from logger import logger
from trainer.ae_trainer import AETrainer
from trainer.sae_trainer import SAETrainer
from trainer.oesae_trainer import OESAETrainer


class TrainerFactory(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.trainer = None

    @classmethod
    def create(cls, model_name, model, train_dataloader, val_dataloader, test_dataloader, tqdm_env="script"):
        trainer_factory = cls(model_name=model_name)

        trainer = None

        if model_name == "ae":
            trainer = AETrainer(
                model=model,
                model_file_metadata=ModelFileMetadata(model_name=model_name),
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                hyperparameters=cls.get_hyperparameters(model_name=model_name),
                tqdm_env=tqdm_env
            )
        elif model_name == "oesae":
            trainer = OESAETrainer(
                model=model,
                model_file_metadata=ModelFileMetadata(model_name=model_name),
                train_loader=train_dataloader[0],
                outlier_loader=train_dataloader[1],
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                hyperparameters=cls.get_hyperparameters(model_name=model_name),
                tqdm_env=tqdm_env
            )
        elif model_name == "model_1":
            pass

        # Set
        trainer_factory.trainer = trainer

        return trainer_factory

    @classmethod
    def get_hyperparameters(cls, model_name):
        hyperparameters = None

        if model_name == "cnn_custom":
            hyperparameters = Hyperparameters(
                optimizer_cls=optim.Adam,
                criterion=F.binary_cross_entropy,
                n_epoch=5,
                lr=1e-3,
                hypothesis_threshold=0.5,
                weight_decay=0,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        elif model_name == "sae":
            hyperparameters = Hyperparameters(
                optimizer_cls=optim.Adam,
                criterion=F.mse_loss,
                n_epoch=100,
                lr=1e-3,
                weight_decay=0,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        elif model_name == "model_1":
            pass

        return hyperparameters

    def do(self, mode):
        logger.info(f"Start to {mode}")
        result_dict = dict()

        if mode == "train":
            result_dict = self.trainer.train()
        elif mode == "predict":
            result_dict = self.trainer.predict()

        logger.info(f"Completed to {mode}")
        return result_dict
