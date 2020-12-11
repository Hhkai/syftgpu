import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import syft as sy
import numpy as np

import time

from hhkdebug import *

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
james = sy.VirtualWorker(id="james", hook=hook)
bill = sy.VirtualWorker(id="bill", hook=hook)

def test_fix():
    a = [[10.55, 20.6, 30.5, 4.1], [-1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    b = [[40.5, 30.5, 20.8, 1.2], [40, 3, 2, 1], [40, 3, 2, 1], [40, 3, 2, 1]]
    # a = [0, 0, 0 , 0, 0, 0, 0, 0]
    # b = [0, 0, 0 , 0, 0, 0, 0, 0]
    # aa = torch.tensor(a).fix_precision(dtype="int24").share(bob, alice, crypto_provider=bill)
    # bb = torch.tensor(b).fix_precision(dtype="int24").share(bob, alice, crypto_provider=bill)

    aa = torch.tensor(a).fix_precision().share(bob, alice, crypto_provider=bill)
    bb = torch.tensor(b).fix_precision().share(bob, alice, crypto_provider=bill)
    # print("while begin:")
    #while True:
    for i in range(2):
        c = aa.mm(bb)
        #c = aa + bb

        print(c.get().float_precision())
        print(np.array(a) @ np.array(b))


if __name__ == '__main__':
    test_fix()