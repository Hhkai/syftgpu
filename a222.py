from torch import tensor

from a111 import a
from follow import x

if __name__ == '__main__':
    print((a == x().cuda()).all())