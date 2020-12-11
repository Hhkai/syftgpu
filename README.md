# syftgpu
forked from OpenMined/PySyft

syft==0.2.9
pytorch==1.4.0
python=3.7

syftgpu到此告一段落了
实现了snn在gpu上的加速
但是由于字长的限制, 矩阵规模大了之后addmm会导致加法溢出浮点精度 

能跑玩具模型, 能做基本运算
