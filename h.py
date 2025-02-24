import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
dwt = DWT1DForward(wave='db6', J=3)
X = torch.randn(10, 5, 100)
yl, yh = dwt(X)
print(yl.shape)
