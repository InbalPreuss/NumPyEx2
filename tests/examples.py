import numpy as np


def quadratic(x):
    return x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2


def example_quadratic():
    ineq_constraints = [lambda x: - x[0],  # x >= 0
                        lambda x: - x[1],  # y >= 0
                        lambda x: - x[2]]  # z >= 0
    eq_constraints_mat = np.array([1, 1, 1])  # x+y+z
    eq_constraints_rhs = np.array([1])  # = 1
    x0 = np.array([0.1, 0.2, 0.7])
    return quadratic, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0


def linear(x):
    return -1 * x[0] + (-1) * x[1]


def example_linear():
    ineq_constraints = [lambda x: (-1) * x[1],  # y >= 0
                        lambda x: x[0] - 2,  # x <= 2
                        lambda x: x[1] - 1,  # y <= 1
                        lambda x: (-1) * x[1] + (-1) * x[0] + 1]  # y + x >= 1
    eq_constraints_mat = None
    eq_constraints_rhs = None
    x0 = np.array([0.5, 0.75])
    return linear, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0
