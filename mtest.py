import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import syft as sy
import numpy as np

import time

from hhkdebug import *
import datas
from follow import fc1w, x 

hook = sy.TorchHook(torch)
#torch.set_printoptions(threshold=200)

alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
james = sy.VirtualWorker(id="james", hook=hook)
bill = sy.VirtualWorker(id="bill", hook=hook)

def mul(a, b):
    # print(a.shape, b.shape)
    a = a.share(bob, alice, crypto_provider=bill)
    b = b.share(bob, alice, crypto_provider=bill)
    c = a.matmul(b)
    print(c.get())

def test_fix():
    a = x()    # 128 784
    b = fc1w()     # 128 784

    row = 196

    a = a.t()[:row].t()
    b = b.t()[:row]

    aaa = a[0:1]
    bbb = b.t()[10:11].t()
    print(aaa, bbb)
    mul(aaa, bbb)

    a6 = a * 1000
    b6=b * 1000
    a7=a.long()
    b7=b.long()

    mul(a6,b6)
    mul(a7,b7)


def ab():
    # -5893392
    a = datas.a
    b= datas.b

    for i in range(100): mul(a, b)
    mul(b,a)

def aaa():
    a = [122779]
    b = [-48]
    a = tensor(a)
    b = tensor(b)
    for i in range(10): mul(a, b)

if __name__ == '__main__':
    # test_fix()
    ab()
    # aaa()
