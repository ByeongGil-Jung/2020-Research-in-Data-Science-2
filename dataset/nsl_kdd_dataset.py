import os

import pandas as pd
import torch
import numpy as np

from dataset.base import DatasetBase
from properties import APPLICATION_PROPERTIES
from logger import logger


class NSLKDDDataFrameFactory(object):

    NSLKDD_DATASET_HOME_DIR_PATH = os.path.join(APPLICATION_PROPERTIES.DATA_DIRECTORY_PATH, "NSL-KDD")

    TRAIN_TXT_FILE_PATH = os.path.join(NSLKDD_DATASET_HOME_DIR_PATH, "KDDTrain+.txt")
    TEST_TXT_FILE_PATH = os.path.join(NSLKDD_DATASET_HOME_DIR_PATH, "KDDTest+.txt")

    TRAIN_20_PERCENT_TXT_FILE_PATH = os.path.join(NSLKDD_DATASET_HOME_DIR_PATH, "KDDTrain+_20Percent.txt")
    TEST_21_TXT_FILE_PATH = os.path.join(NSLKDD_DATASET_HOME_DIR_PATH, "KDDTest-21.txt")

    COLUMN_NAME_LIST = ["duration", "protocol_type", "service", "flag", "src_bytes",
                        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    CATEGORY_CLASS_LIST = ["protocol_type", "service", "flag"]

    CLASS_INDEX_DICT = {
        'normal': 0,
        'guess_passwd': 1,
        'xterm': 2,
        'spy': 3,
        'xsnoop': 4,
        'pod': 5,
        'sendmail': 6,
        'snmpgetattack': 7,
        'multihop': 8,
        'processtable': 9,
        'teardrop': 10,
        'rootkit': 11,
        'ipsweep': 12,
        'saint': 13,
        'worm': 14,
        'land': 15,
        'named': 16,
        'sqlattack': 17,
        'xlock': 18,
        'udpstorm': 19,
        'portsweep': 20,
        'neptune': 21,
        'ftp_write': 22,
        'ps': 23,
        'smurf': 24,
        'perl': 25,
        'mscan': 26,
        'apache2': 27,
        'imap': 28,
        'phf': 29,
        'mailbomb': 30,
        'warezmaster': 31,
        'snmpguess': 32,
        'satan': 33,
        'nmap': 34,
        'back': 35,
        'warezclient': 36,
        'buffer_overflow': 37,
        'loadmodule': 38,
        'httptunnel': 39
    }

    @classmethod
    def create(cls, dataset_name):
        def __create_raw(dataset_name):
            dataset_name = dataset_name.lower()
            csv_file_path = ""

            # 0102
            if dataset_name == "train":
                csv_file_path = cls.TRAIN_TXT_FILE_PATH
            elif dataset_name == "test":
                csv_file_path = cls.TEST_TXT_FILE_PATH
            elif dataset_name == "train_20_percent":
                csv_file_path = cls.TRAIN_20_PERCENT_TXT_FILE_PATH
            elif dataset_name == "test_21":
                csv_file_path = cls.TEST_21_TXT_FILE_PATH

            # Init dataframe
            df = pd.read_csv(csv_file_path, sep=",", index_col=False, names=cls.COLUMN_NAME_LIST)

            logger.info(f"Success to load dataset dataframe : {dataset_name}")

            return df

        # Create
        if dataset_name == "entire":
            train_df = __create_raw(dataset_name="train")
            test_df = __create_raw(dataset_name="test")

            df = pd.concat([train_df, test_df], axis=0)
            df = df.reset_index()

            logger.info(f"Success to load dataset dataframe : {dataset_name}")
        else:
            df = __create_raw(dataset_name=dataset_name)

        return df

    @classmethod
    def oe_preprocess(cls, dataframe, abnormal_sample_ratio, normal_abnormal_ratio, val_test_ratio, is_shuffle):
        df = dataframe.copy()

        y_data = df.pop("label")
        y_data = y_data.replace(cls.CLASS_INDEX_DICT)

        # Split normal and abnormal
        normal_y_data = y_data[y_data == 0]
        abnormal_y_data = y_data[y_data > 0]

        normal_y_data_index_list = normal_y_data.index.to_list()
        abnormal_y_data_index_list = abnormal_y_data.index.to_list()
        abnormal_outlier_y_data_index_list = list()

        if is_shuffle:
            np.random.shuffle(normal_y_data_index_list)
            np.random.shuffle(abnormal_y_data_index_list)

        normal_x_data = df.iloc[normal_y_data_index_list]
        normal_y_data = y_data.iloc[normal_y_data_index_list]

        # Sample abnormal data
        if abnormal_sample_ratio:
            abnormal_sample_size = int(abnormal_sample_ratio * len(abnormal_y_data_index_list)) * 2
            abnormal_y_data_entire_index_list = np.random.choice(abnormal_y_data_index_list,
                                                                 size=abnormal_sample_size, replace=False)

            abnormal_y_data_index_list = abnormal_y_data_entire_index_list[:abnormal_sample_size // 2]
            abnormal_outlier_y_data_index_list = abnormal_y_data_entire_index_list[abnormal_sample_size // 2:]

        abnormal_x_data = df.iloc[abnormal_y_data_index_list]
        outlier_x_data = df.iloc[abnormal_outlier_y_data_index_list]

        abnormal_y_data = y_data.iloc[abnormal_y_data_index_list]
        outlier_y_data = y_data.iloc[abnormal_outlier_y_data_index_list]

        test_normal_data_size = int(len(abnormal_x_data) * normal_abnormal_ratio)

        x_train = normal_x_data[test_normal_data_size:]
        y_train = normal_y_data[test_normal_data_size:]

        normal_x_test = normal_x_data[:test_normal_data_size]
        normal_y_test = normal_y_data[:test_normal_data_size]

        x_test = pd.concat([normal_x_test, abnormal_x_data], axis=0)
        y_test = pd.concat([normal_y_test, abnormal_y_data], axis=0)

        logger.info(f"Entire data size : {len(df)}")
        logger.info(f"Normal data size : {len(normal_x_data)}")
        logger.info(f"Abnormal data size : {len(abnormal_x_data)}")
        logger.info(f"Outlier data size : {len(outlier_x_data)}")
        logger.info(f"Normal test data size : {len(normal_x_test)}")
        logger.info(f"Training data size : {len(x_train)}")
        logger.info(f"Test data size : {len(x_test)}")

        # Split numerical and categorical
        numerical_class_list = [column for column in df.columns if column not in cls.CATEGORY_CLASS_LIST]

        numerical_x_train = x_train[numerical_class_list]
        numerical_x_test = x_test[numerical_class_list]
        numerical_outlier_x_data = outlier_x_data[numerical_class_list]

        categorical_x_train = x_train[cls.CATEGORY_CLASS_LIST]
        categorical_x_test = x_test[cls.CATEGORY_CLASS_LIST]
        categorical_outlier_x_data = outlier_x_data[cls.CATEGORY_CLASS_LIST]

        categorical_x_entire = pd.concat([categorical_x_train, categorical_x_test, categorical_outlier_x_data],
                                         axis=0)
        categorical_x_entire = pd.get_dummies(data=categorical_x_entire)

        start_size_pivot = 0
        end_size_pivot = len(categorical_x_train)
        categorical_x_train = categorical_x_entire.iloc[start_size_pivot:end_size_pivot, :]

        start_size_pivot += len(categorical_x_train)
        end_size_pivot += len(categorical_x_test)
        categorical_x_test = categorical_x_entire.iloc[start_size_pivot:end_size_pivot, :]

        start_size_pivot += len(categorical_x_test)
        end_size_pivot += len(categorical_outlier_x_data)
        categorical_outlier_x_data = categorical_x_entire.iloc[start_size_pivot:end_size_pivot, :]

        x_train = pd.concat([numerical_x_train, categorical_x_train], axis=1)
        x_test = pd.concat([numerical_x_test, categorical_x_test], axis=1)
        outlier_x_data = pd.concat([numerical_outlier_x_data, categorical_outlier_x_data], axis=1)

        # Set index
        x_train = x_train.set_index("index")
        x_test = x_test.set_index("index")
        outlier_x_data = outlier_x_data.set_index("index")

        del categorical_x_entire

        return x_train, y_train, x_test, y_test, outlier_x_data, outlier_y_data

    @classmethod
    def preprocess(cls, dataframe, abnormal_sample_ratio, normal_abnormal_ratio, val_test_ratio, is_shuffle):
        df = dataframe.copy()

        y_data = df.pop("label")
        y_data = y_data.replace(cls.CLASS_INDEX_DICT)

        # Split normal and abnormal
        normal_y_data = y_data[y_data == 0]
        abnormal_y_data = y_data[y_data > 0]

        normal_y_data_index_list = normal_y_data.index.to_list()
        abnormal_y_data_index_list = abnormal_y_data.index.to_list()

        if is_shuffle:
            np.random.shuffle(normal_y_data_index_list)
            np.random.shuffle(abnormal_y_data_index_list)

        normal_x_data = df.iloc[normal_y_data_index_list]
        normal_y_data = y_data.iloc[normal_y_data_index_list]

        # Sample abnormal data
        if abnormal_sample_ratio:
            abnormal_sample_size = int(abnormal_sample_ratio * len(abnormal_y_data_index_list))
            abnormal_y_data_entire_index_list = np.random.choice(abnormal_y_data_index_list, size=abnormal_sample_size, replace=False)

            abnormal_y_data_index_list = abnormal_y_data_entire_index_list[:abnormal_sample_size]

        abnormal_x_data = df.iloc[abnormal_y_data_index_list]

        abnormal_y_data = y_data.iloc[abnormal_y_data_index_list]

        test_normal_data_size = int(len(abnormal_x_data) * normal_abnormal_ratio)

        x_train = normal_x_data[test_normal_data_size:]
        y_train = normal_y_data[test_normal_data_size:]

        normal_x_test = normal_x_data[:test_normal_data_size]
        normal_y_test = normal_y_data[:test_normal_data_size]

        x_test = pd.concat([normal_x_test, abnormal_x_data], axis=0)
        y_test = pd.concat([normal_y_test, abnormal_y_data], axis=0)

        logger.info(f"Entire data size : {len(df)}")
        logger.info(f"Normal data size : {len(normal_x_data)}")
        logger.info(f"Abnormal data size : {len(abnormal_x_data)}")
        logger.info(f"Normal test data size : {len(normal_x_test)}")
        logger.info(f"Training data size : {len(x_train)}")
        logger.info(f"Test data size : {len(x_test)}")

        # Split numerical and categorical
        numerical_class_list = [column for column in df.columns if column not in cls.CATEGORY_CLASS_LIST]

        numerical_x_train = x_train[numerical_class_list]
        numerical_x_test = x_test[numerical_class_list]

        categorical_x_train = x_train[cls.CATEGORY_CLASS_LIST]
        categorical_x_test = x_test[cls.CATEGORY_CLASS_LIST]

        categorical_x_entire = pd.concat([categorical_x_train, categorical_x_test], axis=0)
        categorical_x_entire = pd.get_dummies(data=categorical_x_entire)


        start_size_pivot = 0
        end_size_pivot = len(categorical_x_train)
        categorical_x_train = categorical_x_entire.iloc[start_size_pivot:end_size_pivot, :]

        start_size_pivot += len(categorical_x_train)
        end_size_pivot += len(categorical_x_test)
        categorical_x_test = categorical_x_entire.iloc[start_size_pivot:end_size_pivot, :]

        x_train = pd.concat([numerical_x_train, categorical_x_train], axis=1)
        x_test = pd.concat([numerical_x_test, categorical_x_test], axis=1)

        # Set index
        x_train = x_train.set_index("index")
        x_test = x_test.set_index("index")

        del categorical_x_entire

        return x_train, y_train, x_test, y_test


class NSLKDDDataset(DatasetBase):

    def __init__(self, x, y):
        super(NSLKDDDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        abnormal_label = np.where(self.y.iloc[idx] > 0, 1, 0)
        return torch.tensor(self.x[idx]), torch.tensor(self.y.iloc[idx]), torch.tensor(abnormal_label)
