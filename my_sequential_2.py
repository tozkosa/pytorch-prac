import torch.nn as nn

if __name__ == "__main__":
    model = nn.Sequential()
    model.add_module("conv1", nn.Conv2d(1,20,5))
    model.add_module("relu1", nn.ReLU())
    model.add_module("conv2", nn.Conv2d(20,64,5))
    model.add_module("relu2", nn.ReLU())

    print(model)
    