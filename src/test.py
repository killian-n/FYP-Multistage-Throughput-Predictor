from matplotlib import pyplot as plt
import numpy as np
import random

K = [1,2,3,4,5,6,7,8,9,10]
a = random.sample(range(50), 10)
b = random.sample(range(50), 10)
c = random.sample(range(50), 10)
fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharey=True)
line1 = ax[0,0].plot(K, a, label="a")
line2 = ax[0,0].plot(K, b, label="b")
line3 = ax[0,0].plot(K, c, label="c")
line4 = ax[0,0].plot(K, np.repeat(10, 10), label="average")
ax[0,0].legend(loc="upper right")

ax[0,0].set_title("MSE", fontsize=20)
ax[0,0].set_xlabel("Fold", fontsize=14)
ax[0,0].set_ylabel("MSE", fontsize=14)
plt.show()

print(list(range(1, 11)))