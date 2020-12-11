import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import syft as sy

import time

from hhkdebug import *

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
james = sy.VirtualWorker(id="james", hook=hook)

# A Toy Dataset
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]])
target = torch.tensor([[0],[0],[1],[1.]])


# A Toy Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc2 = nn.Linear(2, 1)
        init.constant(self.fc2.bias, 0.1)
        init.constant(self.fc2.weight, 0.2)

    def forward(self, x):
        x = self.fc2(x)
        return x


protocol = "snn"


if __name__ == '__main__':
    model = Net()

    data = data.fix_precision().share(bob, alice, crypto_provider=james, protocol=protocol, requires_grad=True)
    target = target.fix_precision().share(bob, alice, crypto_provider=james, protocol=protocol, requires_grad=True)
    model = model.fix_precision().share(bob, alice, crypto_provider=james, protocol=protocol, requires_grad=True)

    opt = optim.SGD(params=model.parameters(), lr=0.1).fix_precision()

    t1 = time.time()
    debug = list(model.named_parameters())
    for iter in range(20):
        # 1) erase previous gradients (if they exist)
        t11 = time.time()
        opt.zero_grad()

        # 2) make a prediction
        pred = model(data)

        # 3) calculate how much we missed
        loss = ((pred - target)**2).sum()

        # 4) figure out which weights caused us to miss
        loss.backward()

        # 5) change those weights
        opt.step()
        t12 = time.time()

        # 6) print our progress
        print("loss:%s time:%s " % (loss.get().float_precision(), t12-t11))
    t2 = time.time()
    print(t2-t1)