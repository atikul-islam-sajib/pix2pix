import sys
import logging
import argparse
import os
import joblib
import zipfile
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/dataloader.log",
)

sys.path.append("src/")

from utils import params
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


class Loader:
    """
    The Loader class is designed for loading, processing, and preparing image datasets for training machine learning models. It supports operations like unzipping dataset folders, normalizing images, and creating data loaders for batch processing.

    ## Parameters

    | Parameter   | Type   | Description                                             | Default |
    |-------------|--------|---------------------------------------------------------|---------|
    | dataset     | str    | Path to the zip file containing the dataset.            | None    |
    | normalized  | bool   | Indicates whether to normalize images in the dataset.   | True    |

    ## Attributes

    | Attribute        | Type   | Description                                                         |
    |------------------|--------|---------------------------------------------------------------------|
    | batch_size       | int    | The size of data batches. Loaded from `config["dataloader"]`.       |
    | image_height     | int    | The height of images after resizing. Loaded from `config["dataloader"]`. |
    | image_width      | int    | The width of images after resizing. Loaded from `config["dataloader"]`. |
    | p_value          | float  | Probability value for applying certain transformations. Loaded from `config["dataloader"]`. |
    | normalized_value | float  | The mean and std value used for normalization. Loaded from `config["dataloader"]`. |

    ## Methods

    - `unzip_folder()`: Extracts the dataset zip file into a specified raw data path. Raises an exception if the raw data folder does not exist.

    - `_normalized()`: Returns a composition of transformations for normalizing images if normalization is enabled.

    - `create_dataloader()`: Creates a DataLoader for the dataset. Raises exceptions if raw or processed data folders do not exist.

    ## Examples

    ```python
    from config import config  # Ensure you have a config dictionary defined with necessary dataloader configurations.

    # Initialize the Loader with the path to your dataset zip file.
    loader = Loader(dataset="./data.zip", normalized=True)

    # Unzip the dataset.
    loader.unzip_folder()

    # Create the DataLoader for batch processing.
    dataloader = loader.create_dataloader()
    ```
    """

    def __init__(self, dataset=None, normalized=True):
        self.dataset = dataset
        self.normalized = normalized
        self.batch_size = params()["dataloader"]["batch_size"]
        self.image_height = params()["dataloader"]["image_height"]
        self.image_width = params()["dataloader"]["image_width"]
        self.p_value = params()["dataloader"]["p"]
        self.normalized_value = params()["dataloader"]["normalized"]

    def unzip_folder(self):
        """
        Unzips the dataset contained in the dataset zip file into a predefined raw data directory.

        This method checks if the raw data directory exists before proceeding with the extraction.
        If the directory does not exist, it raises an exception to signal the failure of finding the expected directory.

        Raises:
            Exception: If the raw data folder specified by `RAW_DATA_PATH` does not exist,
                        an exception is raised indicating the folder could not be found.

        Example:
            >>> loader = Loader(dataset="./data.zip")
            >>> loader.unzip_folder()
            This would extract the dataset into the `RAW_DATA_PATH` directory,
            or raise an exception if the directory does not exist.
        """
        with zipfile.ZipFile(self.dataset, "r") as zip_ref:
            if os.path.exists(RAW_DATA_PATH):
                zip_ref.extractall(RAW_DATA_PATH)
            else:
                raise Exception("Could not find the raw data folder".title())

    def _normalized(self):
        """
        Configures and returns a composition of image transformations for normalization and augmentation, applied conditionally based on the `normalized` attribute of the Loader class. This method is intended to prepare images for model training by converting them to tensors, resizing, center cropping, applying random vertical flips, and normalizing pixel values.

        ## Returns

        A torchvision.transforms.Compose object containing a sequence of transformations applied to the images. If the `normalized` attribute is False, this method will return None.

        ## Transformations

        1. **ToTensor**: Converts a PIL image or NumPy `ndarray` into a tensor of the same type with a shape of (C x H x W) in the range [0.0, 1.0].
        2. **Resize**: Resizes the image to the specified `image_height` and `image_width` set in the Loader class.
        3. **CenterCrop**: Crops the given image at the center to the specified `image_height` and `image_width`.
        4. **RandomVerticalFlip**: Randomly flips the image vertically with a probability of `p_value`.
        5. **Normalize**: Normalizes each channel of the image to the `normalized_value` for mean and standard deviation.
        """
        if self.normalized:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((self.image_height, self.image_width)),
                    transforms.CenterCrop((self.image_height, self.image_width)),
                    transforms.RandomVerticalFlip(p=self.p_value),
                    transforms.Normalize(
                        mean=[
                            self.normalized_value,
                            self.normalized_value,
                            self.normalized_value,
                        ],
                        std=[
                            self.normalized_value,
                            self.normalized_value,
                            self.normalized_value,
                        ],
                    ),
                ]
            )

    def create_dataloader(self):
        """
        Creates a DataLoader for the dataset contained within the raw data directory. This DataLoader is configured
        with transformations specified by the `_normalized` method and uses the batch size defined in the class
        attributes. The DataLoader is serialized and saved to the processed data directory for future use.

        Before creating the DataLoader, this method checks for the existence of the raw data directory and the processed
        data directory. If either directory does not exist, an exception is raised.

        Returns:
            DataLoader: A DataLoader object configured for the dataset with the specified transformations and batch size.

        Raises:
            Exception: If the raw data folder specified by `RAW_DATA_PATH` does not exist, or if the processed data folder
                        specified by `PROCESSED_DATA_PATH` does not exist, an exception is raised indicating the missing folder.

        Example:
            >>> loader = Loader(dataset="./data.zip", normalized=True)
            >>> loader.unzip_folder()  # Ensure the dataset is unzipped first
            >>> dataloader = loader.create_dataloader()
            This creates a DataLoader for the unzipped dataset, applies the specified transformations, and saves the
            DataLoader object for later use.

        Note:
            This method relies on global variables `RAW_DATA_PATH` and `PROCESSED_DATA_PATH` to locate the raw and processed
            data directories, respectively. Ensure these variables are correctly set before calling this method.
        """
        if os.path.exists(RAW_DATA_PATH):
            dataset = ImageFolder(root=RAW_DATA_PATH, transform=self._normalized())
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=True
            )

            if os.path.exists(PROCESSED_DATA_PATH):
                joblib.dump(
                    value=dataloader,
                    filename=os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl"),
                )
            else:
                raise Exception("Could not find the processed data folder".title())

            return dataloader
        else:
            raise Exception("Could not find the raw data folder".title())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Loader for the model image-to-image translation using pix2pix".capitalize()
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

    args = parser.parse_args()

    if args.dataset:
        if args.normalized:
            logging.info("Loading the data...".capitalize())
            loader = Loader(dataset=args.dataset, normalized=args.normalized)
            loader.unzip_folder()
            dataloader = loader.create_dataloader()

            logging.info("Data loaded successfully!".capitalize())
    else:
        raise Exception("Please provide a dataset".capitalize())
