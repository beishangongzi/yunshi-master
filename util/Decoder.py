import numpy as np


class Decoder():
    def __init__(self):
        self.colors = [
            [51, 204, 51],
            [255, 204, 0],
            [204, 153, 0],
            [0, 204, 255]
        ]

    def decode(self, x):
        r = np.zeros_like(x)
        g = np.zeros_like(x)
        b = np.zeros_like(x)
        for i in range(1, len(self.colors) + 1):
            mask = x==i
            r[mask] = self.colors[i-1][0]
            g[mask] = self.colors[i-1][1]
            b[mask] = self.colors[i-1][2]
        return np.stack([r, g, b]).transpose((1,2,0))