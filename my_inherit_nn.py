import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2d(x))

if __name__ == "__main__":
    model = Model()
    # model.to('cuda')
    print(model)
