import numpy as np
import torch
import torch.nn.functional as F

from typing import List

class FeatureDecoderLoss(torch.nn.Module):
    def __init__(self,
                 categorical_variables: List[int]=[]):
        super().__init__()
        self.categorical_variables = np.array(categorical_variables)
    
    def forward(self,pred,target):
        if len(self.categorical_variables) > 0:
            cat_loss = F.binary_cross_entropy_with_logits(
                pred[:,self.categorical_variables],
                target[:,self.categorical_variables])
            cont_loss = F.mse_loss(
                pred[:,~self.categorical_variables],
                target[:,~self.categorical_variables])
            loss = cat_loss + cont_loss
        else:
            loss = F.mse_loss(pred,target)
        return loss