import time

from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM
import numpy as np

from trainer.base import OCCTrainerBase

time.time()


class OCSVMTrainer(OCCTrainerBase):

    def __init__(self, model, model_file_metadata, hyperparameters, tqdm_env='script'):
        super(OCSVMTrainer, self).__init__(model, model_file_metadata, hyperparameters, tqdm_env)
        self.model: OneClassSVM
        self.model.nu = hyperparameters.nu
        self.model.kernel = hyperparameters.kernel
        self.model.gamma = hyperparameters.gamma

    def _relabel(self, y_list):
        relabeled_y_list = np.copy(y_list)

        y_0_list = np.where(relabeled_y_list == 0)
        y_1_list = np.where(relabeled_y_list == 1)

        relabeled_y_list[y_0_list] = 1
        relabeled_y_list[y_1_list] = 0

        return relabeled_y_list

    def train(self, x_train):
        self.model.fit(X=x_train)

    def validate(self, x_test, y_test):
        self.model: OneClassSVM
        relabeld_y_test = self._relabel(y_list=y_test)
        x_test_pred = self.model.decision_function(X=x_test)

        fpr, tpr, thresholds = roc_curve(relabeld_y_test, x_test_pred)
        auc_value = auc(fpr, tpr)

        return dict(auc=auc_value)