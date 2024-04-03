import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class ArcFace(nn.Module):
    """
    ArcFace Layer for face recognition.
    This layer is used for face recognition tasks. It computes the class logits using the ArcFace algorithm.

    Attributes:
        emb_size (int): The embedding size (the number of features extracted from CNN).
        num_classes (int): The number of classes.
        s (float): The radius of the projected hypersphere.
        m (float): The arc margin in radians.
    """
    def __init__(self, emb_size, num_classes, s=64.0, m=0.50):
        """
        Constructor for ArcFace Layer.

        Parameters:
            emb_size (int): The embedding size (the number of features extracted from CNN).
            num_classes (int): The number of classes.
            s (float): The radius of the projected hypersphere. Default is 64.0.
            m (float): The arc margin in radians. Default is 0.50.
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
        Computes the forward pass using the Arcface loss and returns the class logits.

        Parameters:
            embedding (torch.Tensor): Extracted embeddings.
            gt (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed class logits through ArcFace Algorithm.
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
        Returns a deep copy of the weights which serve as class centroids.

        Returns:
            torch.Tensor: Deep copy of the weights.
        """
        return self.weights.clone()


class Model(nn.Module):
    """
    Model for face recognition.
    This model is used for projecting an input face image into the target embedding size or the class logits depending
    whether the softmax activation layer will be used or not respectively.
    
    Attributes:
        ft_extractor (torch.nn.Sequential): Extracts meaningful features from the input image.
        num_classes (int): Number of classes in the dataset.
        is_softmax (bool): Indicates whether to use ArcFace Loss (False) or Softmax Loss (True).
        emb_size (int): The target embedding size. Default is 512.
    """
    def __init__(self, ft_extractor, num_classes, is_softmax, emb_size=512):
        """
        Constructor for Model.
        
        Parameters:
            ft_extractor (torch.nn.Sequential): Extracts meaningful features from the input image.
            num_classes (int): Number of classes in the dataset.
            is_softmax (bool): Indicates whether to use Softmax Loss or not.
            emb_size (int): The target embedding size. Default is 512.
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
        Computes the forward pass and performs either of the following:
        1. Returns the extracted embeddings if no softmax activation layer is to be used.
        2. Returns the class logits after applying the softmax if softmax activation layer is to be used.                                        
        
        Parameters:
            input_tensor (torch.Tensor): Input tensor of input face.

        Returns:
            torch.Tensor: Extracted embeddings of emb_size.
        """
        features = torch.squeeze(self.ft_extractor(input_tensor))
        embedding = self.seq(features)
        if self.is_softmax:
            embedding = self.linear(embedding)
        return embedding
