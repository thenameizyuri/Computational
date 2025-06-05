import matplotlib.pyplot as plt
import numpy as np

rsteps = 10000

rar = np.linspace(0.5,4, rsteps)

b = []

for r in rar:
    a = [0.5]
    for it in range(299):
        a.append(r * a[-1] * (1-a[-1]))
    b.append(a[200:])
plt.plot(rar,b)
plt.show()




