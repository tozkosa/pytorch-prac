import torch.nn as nn
from collections import OrderedDict

if __name__ == "__main__":
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1,20,5)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(20,64,5)),
        ('relu2', nn.ReLU())
    ]))

    print(model)
