# Import libraries
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

# Store mean and standard deviations of LFW images aross the RGB channels for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Face_Dataset(Dataset):
    """
    Custom dataset class for face recognition tasks.
    In our case, it will be applied on the LFW dataset.
    
    Attributes:
        images (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of corresponding labels.
        mode (str): Mode of the dataset. "train" for training set and "val" for validation set.
        val_transform (torchvision.transforms.Compose): Composed transformations for validation mode.
        train_transform (torchvision.transforms.Compose): Composed transformations for training mode.
        samples_weight (torch.Tensor): Weighted samples for weighted random sampler.
    """
    def __init__(self, data, mode):
        """
        Constructor for the Face_Dataset class.

        Parameters:
            data (dict): Dictionary containing images and labels.
            mode (str): Mode of the dataset. "train" for training set and "val" for validation set.
        """
        # inherit from base class
        super(Face_Dataset, self).__init__()
        self.images = data["images"]
        self.labels = data["labels"]
        self.mode = mode 
        # Define transformations for validation and training modes
        self.val_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                                 transforms.Normalize(mean=mean, std=std)])

        self.train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
             transforms.Normalize(mean=mean, std=std)])
    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: size of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retrieves the item at the given index.

        Parameters:
            index (int): Index of the item to retrieve.

        Returns:
            torch.Tensor: Image tensor.
            torch.Tensor: Label tensor.
        """
        if self.mode == 'val':
            img = self.val_transform(self.images[index])
            label = torch.tensor(self.labels[index], dtype=torch.int64)
            return img, label
        if self.mode == 'train':
            img = self.train_transform(self.images[index])
            label = torch.tensor(self.labels[index], dtype=torch.int64)
            return img, label

    def create_sampler(self, with_replacement=True):
        """
        Creates Weighted Random Sampler to overcome class imbalance.

        Parameters:
            with_replacement (bool): Allow sampling with replacement.

        Returns:
            torch.utils.data.WeightedRandomSampler: Weighted sampler.
        """
        # get the class frequencies
        class_sample_count = np.array(
            [len(np.where(self.labels == t)[0]) for t in np.unique(self.labels)])
        # get the weights equal to inverse class frequency
        weight = 1. / class_sample_count
        # assign weight = inverse class frequency to each sample
        samples_weight = np.array([weight[t] for t in self.labels.astype("int32")])
        # convert to torch tensor
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=with_replacement)
        self.samples_weight = samples_weight
        return sampler
