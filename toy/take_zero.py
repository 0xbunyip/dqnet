import numpy as np
np.random.seed(123)
a = np.bool_(np.random.randint(0, 2, size = (3, 4, 4)))
print a
b = a[np.arange(1, 1), :, :]
print b
print np.any(b)