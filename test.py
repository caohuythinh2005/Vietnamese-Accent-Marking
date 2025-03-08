import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([[True, False, False],
                  [False, True, True]])
print (a.masked_fill(b == False, -1e9))