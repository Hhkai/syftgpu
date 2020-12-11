import torch
from torch import tensor
import datas


int24field = 2**25
int48field = 2**50
int48max_value = 2**49-1
# def int24module(x):
#     # assert isinstance(x, (torch.Tensor, torch.DoubleTensor)), "%s type error x:%s" % (str(x), type(x))
#     x = ((x + int24field) % int24field).to(device)
#     mask_pos = x > int24max_value
#     if mask_pos.any():
#         mask_pos = mask_pos.type(torch.DoubleTensor).to(device)
#         x = x - (mask_pos * int24field)
#     return x.type(torch.DoubleTensor).to(device)


def int48module(x):
    # assert isinstance(x, (torch.Tensor, torch.DoubleTensor)), "%s type error x:%s" % (str(x), type(x))
    x = ((x + int48field) % int48field)
    mask_pos = x > int48max_value
    if mask_pos.any():
        mask_pos = mask_pos.type(torch.DoubleTensor)
        x = x - (mask_pos * int48field)
    return x.type(torch.DoubleTensor)


def my_cut(x, f=int24field):
    x1 = x // f
    x2 = x % f
    neg_pos = x < 0
    if neg_pos.any():
        nozero_pos = x2 > 0
        nozero_pos = nozero_pos.type(torch.DoubleTensor)
        neg_pos = neg_pos.type(torch.DoubleTensor)
        x1 = x1 - nozero_pos * neg_pos
    return x1, x2


def my_sec_mul(x, y, op_name=None):
    cmd = getattr(torch, "matmul")
    if op_name is not None:
        cmd = getattr(torch, op_name)
    x1, x2 = my_cut(x, 2**12)
    y1, y2 = my_cut(y, 2**12)
    x2y1 = cmd(x2, y1)
    x1y2 = cmd(x1, y2)
    x2y2 = cmd(x2, y2)
    ret = int48module((x2y1 + x1y2) % int24field * int24field + x2y2)
    return ret


def my_mul(x, y, op_name=None):
    """
    x = x // int24 * int24 + x % int24 = x1 * int24 + x2
    x * y = ( x1 * int24 + x2 ) * (y1 * int24 + y2)
        = x1y1 * int48 + (x2y1+x1y2)*int24 + x2*y2"""
    ##
    cmd = getattr(torch, "matmul")
    if op_name is not None:
        cmd = getattr(torch, op_name)
    x1, x2 = my_cut(x)
    y1, y2 = my_cut(y)
    x2y1 = cmd(x2, y1)
    x1y2 = cmd(x1, y2)
    x2y2 = cmd(x2, y2)
    if (x2y2 > int48field).any():
        x2y2 = my_sec_mul(x2, y2)
    ret = int48module((x2y1 + x1y2) % int24field * int24field + x2y2)
    return ret


if __name__ == '__main__':
    x = datas.x.double()
    y = datas.y.double()
    print(x.shape, x.shape[0])
    my_mul(x, y)
    print("hellow ")