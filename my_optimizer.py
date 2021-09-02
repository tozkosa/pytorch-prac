import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(in_features=10, out_features=10, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        return x

def main(opt_conf):
    loss_list = []
    x = torch.randn(1, 10)
    w = torch.randn(1, 1)
    y = torch.mul(w, x) + 2

    net = Net()

    criterion = nn.MSELoss()

    if opt_conf == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.1)
    elif opt_conf == 'momentum_sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    elif opt_conf == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), rho=0.95, eps=1e-04)
    elif opt_conf == 'adagrad':
        optimizer = optim.Adagrad(net.parameters())
    elif opt_conf == 'adam':
        optimizer = optim.Adam(
            net.parameters(), 
            lr=1e-1, 
            betas=(0.9, 0.99),
            eps=1e-09)
    elif opt_conf == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters())

    for epoch in range(20):
        optimizer.zero_grad()
        y_pred = net(x)

        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()

        loss_list.append(loss.data.item())
    
    return loss_list

if __name__ == "__main__":
    loss_dict = {}
    loss_dict["sgd"] = []
    loss_dict["momentum_sgd"] = []
    loss_dict["adadelta"] = []
    loss_dict["adam"] = []
    loss_dict["rmsprop"] = []

    for key, value in loss_dict.items():
        loss_dict[key] = main(key)

    plt.figure()
    plt.plot(loss_dict["sgd"], label='sgd')
    plt.plot(loss_dict["momentum_sgd"], label='momentum_sgd')
    plt.plot(loss_dict["adadelta"], label='adadelta')
    plt.plot(loss_dict["adam"], label='adam')
    plt.plot(loss_dict["rmsprop"], label='rmsprop')
    plt.legend()
    plt.grid()

    plt.show()
    

