from sklearn.svm import OneClassSVM

from logger import logger
from model.cnn_custom import CNNCustom


class ModelFactory(object):

    def __init__(self):
        self.model = None

    @classmethod
    def create(cls, model_name):
        model_factory = cls()

        model = None

        if model_name == "cnn_custom":
            model = CNNCustom(fc_input=19600)
        elif model_name == "model_1":
            pass

        # ML
        if model_name == "ocsvm":
            model = OneClassSVM()

        # Set
        model_factory.model = model

        logger.info(f"Model selected : '{model_name}'")
        return model_factory
