import numpy as np

# Inputs
A = np.array([1., 2., 3.], dtype=np.float32)
B = np.array([
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
], dtype=np.float32)

print(np.cross(A, B))