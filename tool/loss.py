import numpy as np
import matplotlib.pyplot as plt


train_loss, val_loss = [], []
with open('1.txt') as fp:
    lines = fp.readlines()
    for line in lines:
        words = line.split(' ')
        if line.startswith('epoch'):
            word = words[5].strip()
            word = float(word)
            train_loss.append(word)
        else:
            word = words[3].strip()
            word = float(word)
            val_loss.append(word)


plt.figure(figsize=(8, 8))
plt.plot(train_loss, label='train')
plt.plot(val_loss, '--', label='val')
plt.legend()
plt.show()
