import sys
import os
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="logs/trainer.log",
)

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from test import Test


def cli():
    parser = argparse.ArgumentParser(
        description="pix2pix image translation, training and testing".title()
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Define the zip file of the image dataset".capitalize(),
    )
    parser.add_argument(
        "--normalized",
        type=str,
        default=True,
        help="Define if the data is normalized".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1".capitalize())
    parser.add_argument(
        "--lambda_value", type=float, default=100, help="Lambda value".capitalize()
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to run the model".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display the output".capitalize()
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples that is used for plotting".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        if (
            args.dataset
            and args.normalized
            and args.epochs
            and args.lr
            and args.beta1
            and args.lambda_value
            and args.device
            and args.display
        ):
            logging.info("Loading the data...".capitalize())

            loader = Loader(dataset=args.dataset, normalized=args.normalized)
            loader.unzip_folder()
            dataloader = loader.create_dataloader()

            logging.info("Initializing the trainer...".capitalize())

            trainer = Trainer(
                epochs=args.epochs,
                lr=args.lr,
                beta1=args.beta1,
                lambda_value=args.lambda_value,
                device=args.device,
                display=args.display,
            )
            trainer.train(activate=True)

            logging.info("Training completed".capitalize())

    elif args.test:
        if args.samples and args.device:
            logging.info("Testing the model".capitalize())

            plot = Test(num_samples=args.samples, device=args.device)
            plot.test()

            logging.info("Testing completed".capitalize())


if __name__ == "__main__":
    cli()
