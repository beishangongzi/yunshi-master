import torch
from models.FCN import FCN8s, FCN16s, FCN32s


if __name__ == '__main__':
    x = torch.rand((1,3,256,256))
    model = FCN32s(3, 7)
    y = model(x)
    print(y.size())