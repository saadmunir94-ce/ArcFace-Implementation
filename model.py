import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcFace(nn.Module):
    def __init__(self, emb_size, num_classes, s=64.0, m=0.50):
        """
        Constructor for ArcFace Layer
        Args:
            emb_size: int
                the embedding size (the extracted no of features from CNN)
            num_classes: int
                the no of classes
            s: float
                the radius of the projected hypersphere
            m: float
                the arc margin in radians
        """
        # inherit from base class
        super(ArcFace, self).__init__()
        # save configuration
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.s = s
        self.m = m
        # formulate weight tensor
        self.weights = Parameter(torch.FloatTensor(num_classes, emb_size))
        # initialize weights
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embedding, gt):
        """
        Computes the forward pass and returns the class logits
        Args:
            embedding: torch.tensor
                extracted embeddings
            gt: torch.tensor
                groung truth labels

        Returns: the computed class logits through ArcFace Algorithm

        """
        fc7 = F.linear(F.normalize(embedding, dim=1), F.normalize(self.weights, dim=1), bias=None)
        # pick logit at class index
        one_hot = F.one_hot(gt, self.num_classes)
        original_target_logit = fc7[one_hot > 0]
        # theta
        eps = 1e-10
        # clip logits to prevent zero division when backward
        theta = torch.acos(torch.clamp(original_target_logit, -1.0 + eps, 1.0 - eps))
        # marginal_target_logit
        marginal_target_logit = torch.cos(theta + self.m)
        # update fc7
        diff = marginal_target_logit - original_target_logit
        fc7 = fc7 + torch.mul(one_hot, torch.unsqueeze(diff, dim=1))
        # scaling
        fc7 *= self.s
        return fc7

    def get_weights(self):
        """
        Returns a deep copy of the weights which serve as class centroids

        """
        return self.weights.clone()


class Model(nn.Module):
    def __init__(self, ft_extractor, num_classes, is_softmax, emb_size=512):
        """
        Projecting an input face image into the target embedding size
        Args:
            ft_extractor: torch.nn.Sequential
                Extracts meaningful features from the input image
            num_classes: int
                no of classes in the dataset
            is_softmax: Boolean
                "False" means use ArcFace Loss and "True" means use Softmax Loss
            emb_size: int
                the target embedding size. Default = 512
        """
        # inherit from the base class
        super(Model, self).__init__()
        # create the model
        self.ft_extractor = ft_extractor
        self.is_softmax = is_softmax
        self.seq = nn.Sequential(
            nn.BatchNorm1d(num_features=2048),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=emb_size),
            nn.BatchNorm1d(num_features=emb_size)
        )
        if self.is_softmax:
            self.linear = nn.Linear(emb_size, num_classes)

    def forward(self, input_tensor):
        """
        Computes the forward pass and returns the extracted embeddings for an input face image
        Args:
            input_tensor: torch.tensor
                torch tensor of input face

        Returns: the extracted embeddings of emb_size

        """
        features = torch.squeeze(self.ft_extractor(input_tensor))
        embedding = self.seq(features)
        if self.is_softmax:
            embedding = self.linear(embedding)
        return embedding
