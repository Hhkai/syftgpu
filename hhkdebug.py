import re
from torch import tensor
import syft

pat_alice = re.compile(r'alice:(\d+)]')
pat_bob = re.compile(r'bob:(\d+)]')


def return1():
    return 1


def get_id_of_bob_alice(s):
    if not isinstance(s, str):
        s = str(s)
    bob = pat_bob.findall(s)
    alice = pat_alice.findall(s)
    return zip([int(i) for i in bob], [int(i) for i in alice])


def get_value(bob_dict, alice_dict, keys):
    return [
        bob_dict[key[0]] + alice_dict[key[1]]
        for key in keys
    ]


def see(bob, alice, para_name):
    return get_value(bob._objects, alice._objects, get_id_of_bob_alice(para_name))


def s(q):
    assert isinstance(q, str)
    bob = syft.local_worker.get_worker('bob')
    alice = syft.local_worker.get_worker('alice')
    return see(bob, alice, q)


if __name__ == '__main__':
    s = '''[('fc1.weight', Parameter containing:
(Wrapper)>AutogradTensor>FixedPrecisionTensor>[AdditiveSharingTensor]
	-> [PointerTensor | me:50361085085 -> bob:60854837954]
	-> [PointerTensor | me:46392042374 -> alice:86280671772]
	*crypto provider: james*), ('fc1.bias', Parameter containing:
(Wrapper)>AutogradTensor>FixedPrecisionTensor>[AdditiveSharingTensor]
	-> [PointerTensor | me:86341040135 -> bob:36977700710]
	-> [PointerTensor | me:9080366468 -> alice:90742283460]
	*crypto provider: james*), ('fc2.weight', Parameter containing:
(Wrapper)>AutogradTensor>FixedPrecisionTensor>[AdditiveSharingTensor]
	-> [PointerTensor | me:33809129498 -> bob:21520058680]
	-> [PointerTensor | me:92061425377 -> alice:32866728916]
	*crypto provider: james*), ('fc2.bias', Parameter containing:
(Wrapper)>AutogradTensor>FixedPrecisionTensor>[AdditiveSharingTensor]\t
	-> [PointerTensor | me:93322073019 -> bob:85299420635]
	-> [PointerTensor | me:29982049495 -> alice:54553083986]
	*crypto provider: james*)]'''
    print(get_id_of_bob_alice(s))