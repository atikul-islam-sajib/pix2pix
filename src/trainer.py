import sys
import os
import logging
import argparse
import yaml
import torch
import torch.optim as optim
from torchvision.utils import save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="logs/trainer.log",
)

sys.path.append("src/")

from utils import (
    weight_init,
    device_init,
    params,
    load_pickle,
    ignore_warnings,
    saved_config,
    clean,
)
from config import (
    PROCESSED_DATA_PATH,
    TRAIN_CHECKPOINTS,
    LAST_CHECKPOINTS,
    TRAIN_IMAGES,
)
from generator import Generator
from discriminator import Discriminator


class Trainer:
    """
    A Trainer class for training Generative Adversarial Networks (GANs), specifically designed to
    handle image-to-image translation tasks. This class encapsulates the setup, training, and saving
    functionalities for both generator and discriminator models within the GAN architecture.

    Parameters
    ----------
    epochs : int, default=50
        Number of epochs to train the model.
    lr : float, default=0.0002
        Learning rate for Adam optimizers.
    beta1 : float, default=0.5
        Beta1 hyperparameter for Adam optimizers.
    lambda_value : int, default=50
        Lambda weight for L1 loss in generator loss computation.
    device : str, default="mps"
        Device on which the model will be trained ("cpu", "cuda", "mps").
    display : bool, default=True
        If True, training progress is printed to stdout; otherwise, it's logged to a file.

    Attributes
    ----------
    dataloader : DataLoader
        The DataLoader instance holding the training data.
    device : torch.device
        Computed device based on user input and availability.
    netG : Generator
        The generator model.
    netD : Discriminator
        The discriminator model.
    optimizerG : torch.optim.Optimizer
        Optimizer for the generator.
    optimizerD : torch.optim.Optimizer
        Optimizer for the discriminator.
    criterion : torch.nn.modules.loss
        Loss function for computing model losses.

    Examples
    --------
    >>> trainer = Trainer(epochs=20, lr=0.0002, beta1=0.5, lambda_value=100, device="cuda", display=True)
    >>> trainer.train(activate=True)

    This will initialize the training process for the model with the specified parameters, for 20 epochs,
    using a learning rate of 0.0002, beta1 of 0.5, lambda_value of 100, on the CUDA device with display enabled.
    """

    def __init__(
        self,
        epochs=50,
        lr=0.0002,
        beta1=0.5,
        lambda_value=50,
        device="mps",
        display=True,
        clean_folder=True,
    ):
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.beta1 = beta1
        self.lambda_value = lambda_value
        self.critics = 2
        self.steps = 1000
        self.images = 64
        self.display = display
        self.is_clean = clean_folder

    def __setup__(self, activate=False):
        """
        Sets up the training environment by initializing the data loader, device, models, optimizers, and loss criterion.
        This method is meant to be called internally by the `train` method if activation is set to True.

        Parameters
        ----------
        activate : bool, default=False
            If True, the setup process is initiated. Otherwise, a ValueError is raised.

        Raises
        ------
        ValueError
            If `activate` is not set to True, indicating that the setup should not proceed.

        Notes
        -----
        This method is private and should not be called directly by users.
        """
        if activate == True:
            self.dataloader = load_pickle(
                path=os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl")
            )
            self.device = device_init(device=self.device)

            self.netG = Generator().to(self.device)
            self.netD = Discriminator().to(self.device)

            self.netG.apply(weight_init)
            self.netD.apply(weight_init)

            self.optimizerG = optim.Adam(
                self.netG.parameters(),
                lr=self.lr,
                betas=(self.beta1, params()["model"]["beta2"]),
            )
            self.optimizerD = optim.Adam(
                self.netD.parameters(),
                lr=self.lr,
                betas=(self.beta1, params()["model"]["beta2"]),
            )

            self.criterion = torch.nn.BCELoss()

        else:
            raise ValueError("Activation is not set to True".capitalize())

    def train_discriminator(self, **kwargs):
        """
        Trains the discriminator model using both real and fake data batches.
        This involves computing the loss for real and generated images separately and then updating the discriminator's weights.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing 'realA' and 'realB', the input and target images for the discriminator.

        Returns
        -------
        loss : torch.Tensor
            The total loss for the discriminator after processing both real and fake images.

        Notes
        -----
        This method is designed to be called within the main training loop in the `train` method.
        """
        self.optimizerD.zero_grad()

        inputs_real = torch.cat((kwargs["realA"], kwargs["realB"]), 1).to(self.device)
        real_predict = self.netD(inputs_real)
        real_loss = 0.5 * self.criterion(
            real_predict, torch.ones(real_predict.size()).to(self.device)
        )
        real_loss.backward()

        fakeB = self.netG(kwargs["realA"]).detach()
        inputs_fake = torch.cat((kwargs["realA"], fakeB), 1).to(self.device)
        fake_predict = self.netD(inputs_fake)
        fake_loss = 0.5 * self.criterion(
            fake_predict, torch.zeros(fake_predict.size()).to(self.device)
        )
        fake_loss.backward()

        self.optimizerD.step()

        return real_loss + fake_loss

    def train_generator(self, **kwargs):
        """
        Trains the generator model by generating images and then using the discriminator to evaluate them.
        The generator's loss is calculated based on the discriminator's predictions and an L1 loss between the generated and real images.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing 'realA' and 'realB', the input and target images for the generator.

        Returns
        -------
        loss : torch.Tensor
            The total loss for the generator after generating images and evaluating them with the discriminator.

        Notes
        -----
        This method is designed to be called within the main training loop in the `train` method, typically multiple times per discriminator update to maintain the balance between the generator and discriminator.
        """
        self.optimizerG.zero_grad()

        generatedB = self.netG(kwargs["realA"])
        inputs_generated = torch.cat((kwargs["realA"], generatedB), 1).to(self.device)
        generated_predict = self.netD(inputs_generated)
        generated_loss = (
            self.criterion(
                generated_predict, torch.ones(generated_predict.size()).to(self.device)
            )
            + self.lambda_value * torch.abs(generatedB - kwargs["realB"]).sum()
        )

        generated_loss.backward()

        self.optimizerG.step()

        return generated_loss

    def show_progress(self, **kwargs):
        """
        Displays or logs the training progress, including the current epoch, step, discriminator loss, and generator loss.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing 'epoch', 'index', 'D_loss', and 'G_loss', providing context for the current training progress.

        Notes
        -----
        The output destination (stdout or log file) is determined by the `display` attribute of the Trainer class.
        """
        if self.display:
            print(
                "[Epochs - {}/{}] - [Steps - {}/{}] - [D_Loss - {}] - [G_Loss - {}]".format(
                    kwargs["epoch"] + 1,
                    self.epochs,
                    kwargs["index"] + 1,
                    len(self.dataloader),
                    sum(kwargs["D_loss"]) / len(kwargs["D_loss"]),
                    sum(kwargs["G_loss"]) / len(kwargs["G_loss"]),
                )
            )
        else:
            logging.info(
                "[Epochs - {}/{}] - [Steps - {}/{}] - [D_Loss - {}] - [G_Loss - {}]".format(
                    kwargs["epoch"] + 1,
                    self.epochs,
                    kwargs["index"] + 1,
                    len(self.dataloader),
                    sum(kwargs["D_loss"]) / len(kwargs["D_loss"]),
                    sum(kwargs["G_loss"]) / len(kwargs["G_loss"]),
                )
            )

    def save_models(self, **kwargs):
        """
        Saves the generator model's state dict either at specified checkpoints or at the end of training.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing 'epoch', which is used to determine the save path and naming convention for the model checkpoints.

        Raises
        ------
        Exception
            If the checkpoints path is not found, indicating an issue with the specified directory paths for saving model weights.

        Notes
        -----
        This method aims to ensure model progress is saved periodically and at the end of training, facilitating model evaluation and continuation of training if needed.
        """
        if (kwargs["epoch"] + 1) != self.epochs:
            if os.path.exists(TRAIN_CHECKPOINTS):
                torch.save(
                    self.netG.state_dict(),
                    os.path.join(
                        TRAIN_CHECKPOINTS, "netG_{}.pth".format(kwargs["epoch"] + 1)
                    ),
                )
            else:
                raise Exception("Checkpoints path (train) is not found".capitalize())
        else:
            if os.path.exists(LAST_CHECKPOINTS):
                torch.save(
                    self.netG.state_dict(),
                    os.path.join(LAST_CHECKPOINTS, "last_netG.pth"),
                )
            else:
                raise Exception("Checkpoints path (last) is not found".capitalize())

    def train(self, activate=False):
        """
        Initiates the training process for the Generative Adversarial Network (GAN). This method setups
        the training environment, iterates over the dataset for the specified number of epochs, and trains
        the discriminator and generator models.

        Before training begins, the method calls the private `__setup__` method to initialize models,
        optimizers, and other training components if `activate` is True. It also prepares the device
        (CPU/GPU/MPS) for training. Throughout the training process, this method manages the alternation
        between training the discriminator and generator, computes losses, updates models' weights,
        and optionally displays or logs the training progress. Finally, it saves the generator model's
        checkpoints at specified intervals and the last epoch.

        Parameters
        ----------
        activate : bool, default=False
            If True, activates the training setup process. Must be set to True to start training.

        Raises
        ------
        ValueError
            If `activate` is False, indicating the training setup was not initialized.

        Examples
        --------
        >>> trainer = Trainer(epochs=20, lr=0.0002, beta1=0.5, lambda_value=100, device="cuda", display=True)
        >>> trainer.train(activate=True)

        This example demonstrates initiating the training process with previously set parameters.
        The training will proceed for 20 epochs, using an Adam optimizer with a learning rate of 0.0002
        and beta1 of 0.5. The model will be trained on the CUDA device with progress displayed on stdout.

        Note
        ----
        - This method is designed to handle exceptions gracefully by logging any encountered errors during
          the model saving phase and continues the training process.
        - The method assumes a previously prepared DataLoader and models (Generator and Discriminator) which
          are setup and passed through the `__setup__` method when `activate=True`.
        """
        self.__setup__(activate=activate)
        clean(activate=self.is_clean)
        ignore_warnings()

        for epoch in range(self.epochs):
            D_loss = list()
            G_loss = list()
            for index, (images, _) in enumerate(self.dataloader):
                realA = images[:, :, :, :256].to(self.device)
                realB = images[:, :, :, 256:].to(self.device)

                d_loss = self.train_discriminator(realA=realA, realB=realB)
                D_loss.append(d_loss.item())

                for _ in range(self.critics):
                    g_loss = self.train_generator(realA=realA, realB=realB)
                    G_loss.append(g_loss.item())

                if (index + 1) % self.steps == 0:
                    self.show_progress(
                        epoch=epoch, index=index, D_loss=D_loss, G_loss=G_loss
                    )

            try:
                self.save_models(epoch=epoch)
            except Exception as e:
                logging.info(
                    "The error caught in the section - {}".format(e).capitalize()
                )
            else:

                image, _ = next(iter(self.dataloader))
                realA = image[:, :, :, :256].to(self.device)
                targets = self.netG(realA)

                save_image(
                    targets,
                    os.path.join(TRAIN_IMAGES, "image_{}.png".format(epoch + 1)),
                    normalize=True,
                    nrow=1,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="To train the model of pix2pix".title()
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
    args = parser.parse_args()
    if args.display:
        logging.info("Display the output".title())

        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            beta1=args.beta1,
            lambda_value=args.lambda_value,
            device=args.device,
            display=args.display,
        )

        train_config = {
            "train": {
                "epochs": args.epochs,
                "lr": args.lr,
                "beta1": args.beta1,
                "lambda_value": args.lambda_value,
                "device": args.device,
                "display": args.display,
            }
        }
        logging.info("Training the model".title())
        trainer.train(activate=True)

        saved_config(config_file=train_config, filename="./train_params.yml")

        logging.info("Training completed".title())
    else:
        raise ValueError(
            "Define the arguments to train the model properly".capitalize()
        )
