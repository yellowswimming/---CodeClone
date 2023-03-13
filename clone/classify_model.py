import torch
import torch.nn as nn

class ClassifyModel(nn.Module):
    def __init__(self, model1, model2):
        super(ClassifyModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, input1, input2):
        self.model1.batch_size = len(input1)
        self.model2.batch_size = len(input1)
        classify1 = self.model1(input1)
        _, predicted1 = torch.max(classify1.data, 1)
        classify2 = self.model1(input2)
        _, predicted2 = torch.max(classify2.data, 1)
        equal_mask = torch.eq(predicted1, predicted2)

        # 获取相等位置的下标
        equal_indices = equal_mask.nonzero()

        # 将相等位置的值设置为True，其他位置的值设置为False
        result = torch.zeros(len(predicted1), dtype=bool)
        result[equal_indices] = True
        result = result.unsqueeze(dim=1).numpy()

        return result