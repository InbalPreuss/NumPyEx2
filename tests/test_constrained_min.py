import unittest
import numpy as np
from matplotlib import pyplot as plt

from src.constrained_min import LogBarrier
from examples import *
from src.utils import *


class TestConstrainedMin(unittest.TestCase):
    def test(self, test_name, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
        log_barrier = LogBarrier(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        solution, path, objectives = log_barrier.interior_pt(test_name)

        return solution, path, objectives

    def test_qp(self):
        quadratic, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = example_quadratic()
        solution, path, objectives = self.test('qp', quadratic, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)

        final_objective = quadratic(solution)
        final_constraints = [constr(solution) for constr in ineq_constraints]

        # The final candidate
        print(f"The end point is = {solution}")
        # Objective and constraint values at the final candidate
        print(f"The final objective value is = {final_objective}")
        print(f"The final constraint values are = {final_constraints}")

        plot_feasible_region_qp(path)
        plot_feasible_region_qp(path, final_candidate=False)

        plot_obj_vs_iter(objectives)



    def test_lp(self):
        linear, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = example_linear()
        solution, path, objectives = self.test('lp', linear, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)

        final_objective = (-1) * linear(solution)
        final_constraints = [constr(solution) for constr in ineq_constraints]

        # The final candidate
        print(f"The end point is = {solution}")
        # Objective and constraint values at the final candidate
        print(f"The final objective value is = {final_objective}")
        print(f"The final constraint values are = {final_constraints}")
        # # Plot path taken by the algorithm.
        # plot_path(np.array(path))


        # c. Feasible region and the path taken by the algorithm
        plot_polygon(path, solution)
        plot_polygon(path, solution, final_candidate=False)

        # d. Plot the objective value vs. outer iteration number
        objectives = [abs(num) for num in objectives]
        plot_obj_vs_iter(objectives)

        final_constraints = [constr(solution) for constr in ineq_constraints]
        plot_final_candidate(solution, objectives, final_constraints)


if __name__ == "__main__":
    unittest.main()
