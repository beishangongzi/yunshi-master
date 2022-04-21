import torch
from models import *


model = get_model({'name':'FCN32s', 'ch_out':8})
print(model)
x = torch.rand((2, 3,512,512))
y = model(x)
print(y.size())