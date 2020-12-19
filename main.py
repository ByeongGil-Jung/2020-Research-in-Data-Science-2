import argparse

from dataset.factory import DatasetFactory
from logger import logger
from model.factory import ModelFactory
from trainer.factory import TrainerFactory


def main(args):
    model_name = args.model
    data_name = args.data
    mode = args.mode
    tqdm_env = args.tqdm_env

    # Dataset Controller
    dataset_factory = DatasetFactory.create(data_name=data_name)
    train_dataset, val_dataset, test_dataset \
        = dataset_factory.train_dataset, dataset_factory.val_dataset, dataset_factory.test_dataset
    train_dataloader, val_dataloader, test_dataloader \
        = dataset_factory.train_dataloader, dataset_factory.val_dataloader, dataset_factory.test_dataloader

    # Model Controller
    model_factory = ModelFactory.create(model_name=model_name)
    model = model_factory.model

    # Trainer Controller
    trainer_factory = TrainerFactory.create(
        model_name=model_name,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        tqdm_env=tqdm_env
    )

    result_dict = trainer_factory.do(mode=mode)

    print(result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2020-Research-in-Data-Science Experiments Module (AIR Lab, Korea Univ)")
    parser.add_argument("--model", required=False, default="cnn_custom", help="Model name")
    parser.add_argument("--data", required=False, default="kdd-nsl", help="Dataset name")
    parser.add_argument("--mode", required=False, default="train", help="Select the mode, train | predict")
    parser.add_argument("--tqdm_env", required=False, default="script", help="Select the tqdm environment, script | jupyter")
    args = parser.parse_args()

    logger.info(f"Selected parameters : {args}")

    main(args=args)
