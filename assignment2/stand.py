import numpy as np

# x = np.array([0.0, 1.1, 1.2, 0.0, 0.9, 1.2, 1.0, 1.1, 2.2])
# x.shape = (3,3)
N = 3
D = 3
x = np.random.randn(N, D)
# dout = np.array([x*1.0 for x in [1.0, 2.0, 0.0, 4.0, 1.0, 3.0, 0.0, 0.0, 1.0]])
dout = np.random.randn(N, D)
# dout.shape = (3,3)

def loss(x, weight_matrix):
    x_standardization = (x-x.mean(axis=0)) / x.std(axis=0)
    return np.sum(x_standardization*weight_matrix)

def get_derivative(x, weight_matrix, eps):
    dx = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_ = x.copy()
            x_[i][j] = x_[i][j] + eps
            dx[i][j] = (loss(x_, weight_matrix) - loss(x, weight_matrix)) / eps
    return dx

print('empirical derivative')
dx_em = get_derivative(x, dout, eps=1e-8)
print(dx_em)

x_std = np.std(x, axis=0)
x_mean = np.mean(x, axis=0)
x_diff = x - x_mean
x_dout = (x_diff * dout).mean(axis=0)
# dx_ndiagonal_matrix = np.sum(dout, axis=0) * x_intermediate
# # dx_ndiagonal.shape = (1, D)
# # dx_ndiagonal_matrix = dx_ndiagonal.repeat(N, axis=0)
dx = - dout.mean(axis=0) * np.power(x_std, -1) - x_diff * x_dout * np.power(x_std, -3) + dout * np.power(x_std, -1)
print('theoretical derivative')
print(dx)

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

print('relative error: %.2e' % (rel_error(dx, dx_em)))
