import torch as th
from typing import Tuple

from syft.generic.object_storage import device

import syft as sy

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
    x = ((x + int48field) % int48field).to(device)
    mask_pos = x > int48max_value
    if mask_pos.any():
        mask_pos = mask_pos.type(th.DoubleTensor).to(device)
        x = x - (mask_pos * int48field)
    return x.type(th.DoubleTensor).to(device)


def my_cut(x):
    x1 = x // int24field
    x2 = x % int24field
    neg_pos = x < 0
    if neg_pos.any():
        nozero_pos = x2 > 0
        nozero_pos = nozero_pos.type(th.DoubleTensor).to(device)
        neg_pos = neg_pos.type(th.DoubleTensor).to(device)
        x1 = x1 - nozero_pos * neg_pos
    return x1, x2


def my_matmul(x, y):
    """
    x = x // int24 * int24 + x % int24 = x1 * int24 + x2
    x * y = ( x1 * int24 + x2 ) * (y1 * int24 + y2)
        = x1y1 * int48 + (x2y1+x1y2)*int24 + x2*y2"""
    ##
    cmd = getattr(th, "matmul")
    x1, x2 = my_cut(x)
    y1, y2 = my_cut(y)
    x2y1 = cmd(x2, y1)
    x1y2 = cmd(x1, y2)
    x2y2 = cmd(x2, y2)
    ret = (x2y1 + x1y2) % int24field * int24field + x2y2
    ret = int48module(ret)
    return ret

def my_mul(x, y):
    """
    x = x // int24 * int24 + x % int24 = x1 * int24 + x2
    x * y = ( x1 * int24 + x2 ) * (y1 * int24 + y2)
        = x1y1 * int48 + (x2y1+x1y2)*int24 + x2*y2"""
    ##
    cmd = getattr(th, "mul")
    x1, x2 = my_cut(x)
    y1, y2 = my_cut(y)
    x2y1 = cmd(x2, y1)
    x1y2 = cmd(x1, y2)
    x2y2 = cmd(x2, y2)
    return int48module((x2y1 + x1y2) % int24field * int24field + x2y2)


def build_triple(
    op: str,
    shape: Tuple[th.Size, th.Size],
    n_workers: int,
    n_instances: int,
    torch_dtype: th.dtype,
    field: int,
):
    """
    Generates and shares a multiplication triple (a, b, c)

    Args:
        op (str): 'mul' or 'matmul': the op ° which ensures a ° b = c
        shape (Tuple[th.Size, th.Size]): the shapes of a and b
        n_workers (int): number of workers
        n_instances (int): the number of tuples (works only for mul: there is a
            shape issue for matmul which could be addressed)
        torch_dtype (th.dtype): the type of the shares
        field (int): the field for the randomness

    Returns:
        a triple of shares (a_sh, b_sh, c_sh) per worker where a_sh is a share of a
    """
    left_shape, right_shape = shape
    cmd = getattr(th, op)
    low_bound, high_bound = -(field // 2), (field - 1) // 2
    if op == "matmul":
        cmd = my_matmul
    if op == "mul":
        cmd = my_mul
    a = th.randint(low_bound, high_bound, (n_instances, *left_shape), dtype=torch_dtype).to(device)
    b = th.randint(low_bound, high_bound, (n_instances, *right_shape), dtype=torch_dtype).to(device)
    # hhk : some where the field is wrong
    a = int48module(a)
    b = int48module(b)
    

    if op == "mul" and b.numel() == a.numel():
        # examples:
        #   torch.tensor([3]) * torch.tensor(3) = tensor([9])
        #   torch.tensor([3]) * torch.tensor([[3]]) = tensor([[9]])
        if len(a.shape) == len(b.shape):
            c = cmd(a, b)
        elif len(a.shape) > len(b.shape):
            shape = b.shape
            b = b.reshape_as(a)
            c = cmd(a, b)
            b = b.reshape(*shape)
        else:  # len(a.shape) < len(b.shape):
            shape = a.shape
            a = a.reshape_as(b)
            c = cmd(a, b)
            a = a.reshape(*shape)
    else:
        c = cmd(a, b)

    helper = sy.AdditiveSharingTensor(field=field)
    # helper_c = sy.AdditiveSharingTensor(field=field*field)#hhk

    shares_worker = [[0, 0, 0] for _ in range(n_workers)]
    # for i, tensor in enumerate([a, b, c]):
    #     # hhk
    #     if i == 2:
    #         shares = helper_c.generate_shares(secret=tensor, n_workers=n_workers, random_type=torch_dtype)
    #         for w_id in range(n_workers):
    #             shares_worker[w_id][i] = shares[w_id]
    #     else:
    #         shares = helper.generate_shares(secret=tensor, n_workers=n_workers, random_type=torch_dtype)
    #         for w_id in range(n_workers):
    #             shares_worker[w_id][i] = shares[w_id]
    for i, tensor in enumerate([a, b, c]):
        shares = helper.generate_shares(secret=tensor, n_workers=n_workers, random_type=torch_dtype)
        for w_id in range(n_workers):
            shares_worker[w_id][i] = shares[w_id]

    return shares_worker
