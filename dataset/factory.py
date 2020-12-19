import os

from sklearn import preprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset.nsl_kdd_dataset import NSLKDDDataFrameFactory, NSLKDDDataset
from logger import logger
from properties import APPLICATION_PROPERTIES


class DatasetFactory(object):

    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    @classmethod
    def create(cls, data_name):
        dataset_factory = cls()

        train_dataset = None
        val_dataset = None
        test_dataset = None

        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

        if data_name == "nsl_kdd" or data_name == "nsl_kdd_outlier":
            is_used_outlier = True if data_name.split("_")[-1] == "outlier" else False

            if not is_used_outlier:
                x_train, y_train, x_test, y_test = NSLKDDDataFrameFactory.preprocess(
                    dataframe=NSLKDDDataFrameFactory.create("entire"),
                    abnormal_sample_ratio=0.1,
                    normal_abnormal_ratio=1.0,
                    val_test_ratio=0.3,
                    is_shuffle=True
                )

                scaler = preprocessing.StandardScaler()
                scaler.fit(x_train)
                logger.info("Success to fit data with scaling")

                x_train = scaler.transform(x_train.astype(np.float32))
                x_test = scaler.transform(x_test.astype(np.float32))

                train_dataset = NSLKDDDataset(x=x_train, y=y_train)
                test_dataset = NSLKDDDataset(x=x_test, y=y_test)
                logger.info("Success to transform data with scaling")

                train_dataloader = DataLoader(
                    dataset=train_dataset,
                    batch_size=32,
                    shuffle=True,
                    pin_memory=False
                )
                test_dataloader = DataLoader(
                    dataset=test_dataset,
                    batch_size=32,
                    shuffle=False,
                    pin_memory=False
                )
            else:
                x_train, y_train, x_test, y_test, outlier_x_data, outlier_y_data = NSLKDDDataFrameFactory.oe_preprocess(
                    dataframe=NSLKDDDataFrameFactory.create("entire"),
                    abnormal_sample_ratio=0.1,
                    normal_abnormal_ratio=1.0,
                    val_test_ratio=0.3,
                    is_shuffle=True
                )

                scaler = preprocessing.StandardScaler()
                scaler.fit(x_train)
                logger.info("Success to fit data with scaling")

                x_train = scaler.transform(x_train.astype(np.float32))
                x_test = scaler.transform(x_test.astype(np.float32))
                outlier_x_data = scaler.transform(outlier_x_data.astype(np.float32))

                train_dataset = NSLKDDDataset(x=x_train, y=y_train)
                test_dataset = NSLKDDDataset(x=x_test, y=y_test)
                outlier_dataset = NSLKDDDataset(x=outlier_x_data, y=outlier_y_data)
                logger.info("Success to transform data with scaling")

                train_dataloader = DataLoader(
                    dataset=train_dataset,
                    batch_size=32,
                    shuffle=True,
                    pin_memory=False
                )
                test_dataloader = DataLoader(
                    dataset=test_dataset,
                    batch_size=32,
                    shuffle=False,
                    pin_memory=False
                )
                outlier_dataloader = DataLoader(
                    dataset=outlier_dataset,
                    batch_size=32,
                    shuffle=True,
                    pin_memory=False
                )

                train_dataset = (train_dataset, outlier_dataset)
                train_dataloader = (train_dataloader, outlier_dataloader)
        elif data_name == "data_1":
            pass

        # Set
        dataset_factory.train_dataset = train_dataset
        dataset_factory.val_dataset = val_dataset
        dataset_factory.test_dataset = test_dataset

        dataset_factory.train_dataloader = train_dataloader
        dataset_factory.val_dataloader = val_dataloader
        dataset_factory.test_dataloader = test_dataloader

        logger.info(f"Data selected : '{data_name}'")
        return dataset_factory
