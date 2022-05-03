# Import libraries
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

# Store mean and standard deviations for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Face_Dataset(Dataset):
    def __init__(self, data, mode):
        """
        Constructor of Face_Dataset
        Args:
            data: Python Dictionary
                returns numpy arrays of images and labels with associated keys "images" and "labels"
            mode: str
                "train" means train set
                "val" means validation set
        """
        # inherit from base class
        super(Dataset, self).__init__()
        self.images = data["images"]
        self.labels = data["labels"]
        self.mode = mode  # val or train
        # Compose is the callable class which does chain of transformations on the data
        # Consider creating two different transforms based on whether you are in the training or validation dataset.
        self.val_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                                 transforms.Normalize(mean=mean, std=std)])

        self.train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        """

        Returns:
            length of the dataset
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        used for accessing list items
        Args:
            index: int

        Returns:
            img: torch.tensor
                the corresponding image
            label: torch.tensor
                the corresponding class label

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
        Creates Weighted Random Sampler to overcome class imbalance
        Args:
            with_replacement: Boolean
                Allow sampling with replacement

        Returns:
            sampler: torch.utils.data.WeightedRandomSampler

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
