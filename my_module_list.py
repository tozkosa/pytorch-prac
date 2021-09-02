import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 20, 5), nn.Conv2d(20, 64, 5)])

    def forward(self, x):
        for i, l in enumerate(self.convs):
            x = l(x)
        return x

if __name__ == "__main__":
    model = Model()
    print(model)