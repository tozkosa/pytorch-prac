import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convs = nn.ModuleDict({
            'conv1': nn.Conv2d(1, 20, 5), 
            'conv2':nn.Conv2d(20, 64, 5)})

    def forward(self, x):
        for l in self.convs.values():
            x = l(x)
        return x

if __name__ == "__main__":
    model = Model()
    print(model)