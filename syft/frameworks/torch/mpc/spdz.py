import asyncio
import math
import multiprocessing
import torch as th

import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.generic.utils import allow_command
from syft.generic.utils import remote
from syft.generic.utils import int48module, int48field, int24field, device

from syft.frameworks.torch.mpc.fss import N_CORES

no_wrap = {"no_wrap": True}

def full_name(f):
    return f"syft.frameworks.torch.mpc.spdz.{f.__name__}"


# share level
@allow_command
def spdz_mask(x, y, op: str, dtype: str, torch_dtype: th.dtype, field: int, owner_in):
    """
    Build the shares of delta and epsilon in the SPDZ protocol
    Args:
        x (Tensor): share of x, where the global computation is z = x ° y
        y (Tensor): share of y
        op (str): type of operation ('mul' or 'matmul')
        dtype (str): type of sahres ('int' or 'long')
        torch_dtype (th.dtype): corresponding torch dtype
        field (int): the field of the corresponding AdditiveSharingTensor

    Returns:
        The shares of delta and epsilon
    """
    a, b, c = owner_in.crypto_store.get_keys(
        op=op,
        shapes=(x.shape, y.shape),
        n_instances=1,
        remove=False,
        dtype=dtype,
        torch_dtype=torch_dtype,
        field=field,
    )
    return x - a, y - b


def slice(x, j, slice_size):
    x_slice = x[j * slice_size : (j + 1) * slice_size]
    x_slice.owner = x.owner
    return x_slice


def triple_mat_mul(core_id, delta, epsilon, a, b):
    cmd = th.matmul
    delta_b = cmd(delta, b)
    a_epsilon = cmd(a, epsilon)
    delta_epsilon = cmd(delta, epsilon)
    return core_id, int24module(delta_b), int24module(a_epsilon), int24module(delta_epsilon)


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


def my_mul(x, y, op_name=None):
    """
    x = x // int24 * int24 + x % int24 = x1 * int24 + x2
    x * y = ( x1 * int24 + x2 ) * (y1 * int24 + y2)
        = x1y1 * int48 + (x2y1+x1y2)*int24 + x2*y2"""
    ##
    cmd = getattr(th, "matmul")
    if op_name is not None:
        cmd = getattr(th, op_name)
    x1, x2 = my_cut(x)
    y1, y2 = my_cut(y)
    x2y1 = cmd(x2, y1)
    x1y2 = cmd(x1, y2)
    x2y2 = cmd(x2, y2)
    ret = int48module((x2y1 + x1y2) % int24field * int24field + x2y2)
    debug = cmd(x.long().cpu(), y.long().cpu()) % int48field
    debuh = debug - int48field
    debui = ret.long().cpu()
    return ret


# share level
@allow_command
def spdz_compute(j: int, delta, epsilon, op: str, dtype: str, torch_dtype: th.dtype, field: int, owner_in):
    """
    Compute the mul or matmul part of the SPDZ protocol, once delta and epsilon
    have been made public
    Args:
        j (int): the rank of the worker, from 0 to n_worker - 1
        delta (Tensor): delta in the SPDZ protocol
        epsilon (Tensor): epsilon in the SPDZ protocol
        op (str): type of operation ('mul' or 'matmul')
        dtype (str): type of sahres ('int' or 'long')
        torch_dtype (th.dtype): corresponding torch dtype
        field (int): the field of the corresponding AdditiveSharingTensor

    Returns:
        The shares of the result of the multiplication
    """
    a, b, c = owner_in.crypto_store.get_keys(
        op=op,
        shapes=(delta.shape, epsilon.shape),
        n_instances=1,
        remove=True,
        dtype=dtype,
        torch_dtype=torch_dtype,
        field=field,
    )
    gua = False
    if op == "matmul" and gua:

        batch_size = delta.shape[0]

        multiprocessing_args = []
        slice_size = math.ceil(batch_size / N_CORES)
        for core_id in range(N_CORES):
            process_args = (
                core_id,
                slice(delta, core_id, slice_size),
                epsilon,
                slice(a, core_id, slice_size),
                b,
            )
            multiprocessing_args.append(process_args)
        p = multiprocessing.Pool()
        partitions = p.starmap(triple_mat_mul, multiprocessing_args)
        p.close()
        partitions = sorted(partitions, key=lambda k: k[0])
        delta_b = th.cat([partition[1] for partition in partitions])
        a_epsilon = th.cat([partition[2] for partition in partitions])
        delta_epsilon = th.cat([partition[3] for partition in partitions])
    else:
        # cmd = getattr(th, op)
        cmd = my_mul

        delta_b = cmd(delta, b, op)
        a_epsilon = cmd(a, epsilon, op)
        delta_epsilon = cmd(delta, epsilon, op)

        # delta_b = cmd(delta, b)
        # a_epsilon = cmd(a, epsilon)
        # delta_epsilon = cmd(delta, epsilon)

    if j == 0:
        ret = delta_epsilon + delta_b + a_epsilon + c
    else:
        ret = delta_b + a_epsilon + c
    # debug = ret.type(th.int64)
    ret = int48module(ret)
    # debug2 = ret.type(th.int64)
    return ret


def spdz_mul(cmd, x, y, crypto_provider, dtype, torch_dtype, field):
    """Abstractly multiplies two tensors (mul or matmul)
    Args:
        cmd: a callable of the equation to be computed (mul or matmul)
        x (AdditiveSharingTensor): the left part of the operation
        y (AdditiveSharingTensor): the right part of the operation
        crypto_provider (AbstractWorker): an AbstractWorker which is used
            to generate triples
        dtype (str): denotes the dtype of the shares, should be 'long' (default),
            'int' or 'custom'
        torch_dtype (torch.dtype): the real type of the shares, should be th.int64
            (default) or th.int32
        field (int): an integer denoting the size of the field, default is 2**64
    Return:
        an AdditiveSharingTensor
    """

    op = cmd
    locations = x.locations
    # Experimental results don't show real improvements with asynchronous = True
    asynchronous = False  # isinstance(locations[0], WebsocketClientWorker)

    try:
        shares_delta, shares_epsilon = [], []
        for location in locations:
            args = (x.child[location.id], y.child[location.id], op, dtype, torch_dtype, field, location)
            share_delta, share_epsilon = remote(spdz_mask, location=location)(
                *args, return_value=True, return_arity=2
            )
            shares_delta.append(share_delta)
            shares_epsilon.append(share_epsilon)
    except EmptyCryptoPrimitiveStoreError as e:
        if sy.local_worker.crypto_store.force_preprocessing:
            raise
        crypto_provider.crypto_store.provide_primitives(workers=locations, **e.kwargs_)
        return spdz_mul(cmd, x, y, crypto_provider, dtype, torch_dtype, field)

    delta = sum(shares_delta)
    epsilon = sum(shares_epsilon)

    for location, share_delta, share_epsilon in zip(locations, shares_delta, shares_epsilon):
        location.de_register_obj(share_delta)
        location.de_register_obj(share_epsilon)
        del share_delta
        del share_epsilon

    if not asynchronous:
        shares = []
        for i, location in enumerate(locations):
            args = (th.LongTensor([i]), delta, epsilon, op, dtype, torch_dtype, field, location)
            share = remote(spdz_compute, location=location)(*args, return_value=False)
            shares.append(share)
    else:
        shares = asyncio.run(
            sy.local_worker.async_dispatch(
                workers=locations,
                commands=[
                    (
                        full_name(spdz_compute),
                        None,
                        (th.LongTensor([i]), delta, epsilon, op),
                        {},
                    )
                    for i in [0, 1]
                ],
                return_value=False,
            )
        )

    shares = {loc.id: share for loc, share in zip(locations, shares)}

    response = sy.AdditiveSharingTensor(shares, **x.get_class_attributes())
    return response
