import torch
import numpy as np

a = torch.rand(2, 3)
b = torch.rand(2, 3)

with open("test.txt", "w") as f:
    f.write(np.array2string(torch.flatten(a).numpy(), separator=",") + "\n")
    f.write(np.array2string(torch.flatten(b).numpy(), separator=",") + "\n")

