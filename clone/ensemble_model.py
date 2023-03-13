import numpy as np
import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3, weights):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.weights = weights

    def forward(self, input1, input2):
        self.model1.batch_size = len(input1)
        self.model2.batch_size = len(input1)
        self.model3.batch_size = len(input1)
        output1 = self.model1(input1, input2)
        output2 = self.model2(input1, input2)
        output3 = self.model3(input1, input2)
        ''' print("------------")
        print(output1)
        print(output2)
        print(output3)
        print("------------")
        '''
        classifier1_output = torch.round(output1).bool().cpu().numpy()
        classifier2_output = torch.round(output2).bool().cpu().numpy()
        classifier3_output = output3 # torch.round(output3).bool().cpu().numpy()

        ensemble_output = (classifier1_output * self.weights[0]) + (classifier2_output * self.weights[1]) + (
                classifier3_output * self.weights[2])

        return ensemble_output
