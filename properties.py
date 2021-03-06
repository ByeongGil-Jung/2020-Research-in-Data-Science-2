from dataclasses import dataclass
import os
import pathlib
import random

import numpy as np
import torch

from logger import logger


class Configuration(object):

    DEFAULT_RANDOM_SEED = 777

    @classmethod
    def apply(cls, random_seed=DEFAULT_RANDOM_SEED):
        Configuration.set_torch_seed(random_seed=random_seed)
        Configuration.set_numpy_seed(random_seed=random_seed)
        Configuration.set_python_random_seed(random_seed=random_seed)

        logger.info(f"Complete to apply the random seed, RANDOM_SEED : {random_seed}")

    @classmethod
    def set_torch_seed(cls, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def set_numpy_seed(cls, random_seed):
        np.random.seed(random_seed)

    @classmethod
    def set_python_random_seed(cls, random_seed):
        random.seed(random_seed)


@dataclass
class ApplicationProperties:
    CURRENT_MODULE_PATH = pathlib.Path(__file__).parent.absolute()

    DATA_DIRECTORY_PATH = os.path.join(CURRENT_MODULE_PATH, "data")
    DATASET_DIRECTORY_PATH = os.path.join(CURRENT_MODULE_PATH, "dataset")
    MODEL_DIRECTORY_PATH = os.path.join(CURRENT_MODULE_PATH, "model")
    PREDICT_DIRECTORY_PATH = os.path.join(CURRENT_MODULE_PATH, "predict")
    TRAIN_DIRECTORY_PATH = os.path.join(CURRENT_MODULE_PATH, "trainer")

    DEFAULT_RANDOM_SEED = 777

    DEVICE_CPU = "cpu"

    def __post_init__(self):
        Configuration.apply(random_seed=self.DEFAULT_RANDOM_SEED)


APPLICATION_PROPERTIES = ApplicationProperties()
