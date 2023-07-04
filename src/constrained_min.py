import numpy as np
from scipy.optimize import minimize
from autograd import grad, jacobian
from autograd import hessian
import autograd.numpy as anp

t = 1


class LogBarrier:
    def __init__(self,
                 func,
                 ineq_constraints,
                 eq_constraints_mat,
                 eq_constraints_rhs,
                 x0):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.x0 = x0

    def line_search(self, func, grad, xk, pk, c1=0.01, alpha_init=0.5):
        alpha = alpha_init
        while True:
            func_xk = func(xk)
            func_xk_alpha_pk = func(xk + alpha * pk)
            grad_xk_dot_pk = np.dot(grad.T, pk)

            if func_xk_alpha_pk > func_xk + c1 * alpha * grad_xk_dot_pk:
                alpha *= 0.5
            else:
                return alpha

    def phi(self, x):
        phi_value = 0
        for constraint in self.ineq_constraints:
            if -constraint(x) <= 0:  # if the constraint is violated
                # print("ERROR negative log")
                return np.inf  # returning a large value indicating infeasible solution
            else:
                phi_value -= anp.log(-constraint(x))  # accumulate the log barrier for each constraint

        return phi_value

    def interior_pt(self, test_name, mu=10, eps=1e-5, max_iter=100):
        global t
        objective = lambda x: t * self.func(x) + self.phi(x)

        m = len(self.ineq_constraints)
        path = []
        objectives = []
        p_nt = self.x0

        k = 0
        while m / t >= eps:
            p_nt = self.newton_step(objective=objective, x=p_nt)

            path.append(np.copy(p_nt))
            objectives.append(self.func(p_nt))

            print(f'Iteration: {k} \t x = {p_nt}, f(x) = {(-1) * objectives[k]:.4f}, gap = {m / t:.4f}')
            k += 1
            t *= mu

        return p_nt, path, objectives

    def newton_step(self, objective, x, reg_term=1e-5, max_iter=1000):
        x_p = x
        while max_iter > 0:
            g = self.gradient(x_p)
            h = self.hessian(x_p)

            A = self.eq_constraints_mat
            b = self.eq_constraints_rhs

            # If there are no equality constraints, solve the system of equations to find d
            if A is None:
                p_nt = np.linalg.solve(h, -g)
                lambda_ = np.sqrt(p_nt.T @ h @ p_nt)[0][0]
            else:
                # If there are equality constraints, construct the KKT system
                A = np.reshape(A, (1, -1))
                AT = np.reshape(A.T, (-1, 1))

                # Zero matrix and zero vector
                zeros_m_m = np.zeros((len(b), len(b)))
                zeros_m = np.zeros((len(b), len(b)))

                KKT_mat = np.vstack([np.hstack((h, AT)), np.hstack((A, zeros_m_m))])
                KKT_rhs = np.hstack((np.squeeze(-g), zeros_m.flatten()))

                # Solve the KKT system
                sol = np.linalg.solve(KKT_mat, KKT_rhs)

                # The solution includes the step direction 'd' and the Lagrange multipliers 'lambda_'
                p_nt = sol[:len(x)]
                p_nt = p_nt[:, np.newaxis]

                lambda_ = sol[len(x):][0]

            if (0.5 * lambda_ ** 2) < reg_term:
                return x_p
            else:
                alpha = self.line_search(func=objective, grad=g, xk=x_p, pk=p_nt.T[0])
                if alpha is not None:
                    x_p.shape = (x_p.shape[0], 1)
                    x_n = alpha * p_nt + x_p
                    x_n = np.squeeze(x_n)
                    x_p = np.squeeze(x_n)
                else:
                    print('ERROR')
            max_iter -= 1
        return x_p

    def gradient(self, x):
        fs = np.zeros((x.shape[0], 1))

        grad_func = grad(self.func)
        gx_func = grad_func(x)
        gx_func.shape = (gx_func.shape[0], 1)

        for constraint in self.ineq_constraints:
            grad_func = grad(constraint)
            f = 1 / (-1 * constraint(x)) * grad_func(x)
            f.shape = (x.shape[0], 1)
            fs += f
        return t * gx_func + fs

    def hessian(self, x):
        x = np.squeeze(x)

        grad_func = grad(self.func)
        hessian_j_func = jacobian(grad_func)

        f_1 = np.zeros((x.shape[0], x.shape[0]))
        f_2 = np.zeros((x.shape[0], x.shape[0]))
        for constraint in self.ineq_constraints:
            grad_func = grad(constraint)
            hessian_func = jacobian(grad_func)
            f_1 += 1 / (constraint(x) ** 2) * grad_func(x).reshape(-1, 1) @ grad_func(x).reshape(-1, 1).T
            f_2 += 1 / (-1 * constraint(x)) * hessian_func(x)

        return t * hessian_j_func(x) + f_1 + f_2
