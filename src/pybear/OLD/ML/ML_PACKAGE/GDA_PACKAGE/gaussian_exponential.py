

def gaussian_exponential(x, mu, COVARIANCE_MATRIX):

    x_minus_mu = n.subtract(coordinate, mu)
    exp_term = -.5 * n.matmul(x_minus_mu, n.matmul(n.linalg.inv(SIGMA), n.transpose(x_minus_mu), dtype=float),
                              dtype=float)
    GAUSS_GRID[row_idx][col_idx] = gauss_coeff * n.exp(exp_term)