import torch
from torch.utils.data import TensorDataset
from cl_utils.rded_adapter import RDEDDistiller
x = torch.rand(100, 3, 32, 32)
y = torch.randint(0, 10, (100,))
ds = TensorDataset(x, y)

d = RDEDDistiller(cfg={"iters": 10}, 
                  rded_root=r"C:\Users\ELLEN\Desktop\RDED_CL",
                  main_py_rel="scripts/rded_bridge.py")

synth = d.distill(ds, per_class=5, device="cpu")
print(len(synth), synth[0][0].shape)