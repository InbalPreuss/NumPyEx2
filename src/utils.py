import numpy as np
from matplotlib import pyplot as plt


# Plot the path taken by the algorith
def plot_path(path):
    x = [point[0] for point in path]
    y = [point[1] for point in path]
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.title('Path taken by the algorithm')
    plt.show()


# Plot the objective value vs. outer iteration number
def plot_obj_vs_iter(objectives):
    plt.figure()
    plt.plot(objectives)
    plt.title('Objective value vs. iteration number')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective value')
    plt.show()

def plot_polygon(path, solution, final_candidate=True):
    x = [point[0] for point in path]
    y = [point[1] for point in path]
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.title('Path taken by the algorithm')

    # Define the vertices of the polygon
    vertices = [
        [0, 0],  # (0, 0) - Origin
        [2, 0],  # (2, 0) - x <= 2
        [2, 1],  # (2, 1) - y <= 1
        [1, 1]  # (1, 1) - y + x >= 1
    ]

    # Extract x and y coordinates from the vertices
    x_coords = [vertex[0] for vertex in vertices]
    y_coords = [vertex[1] for vertex in vertices]

    # Plot the polygon
    plt.fill(x_coords, y_coords, alpha=0.3, edgecolor='black', facecolor='blue')
    plt.xlim(-1, 3)  # Set x-axis limits
    plt.ylim(-1, 2)  # Set y-axis limits
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Feasible Region')
    plt.grid(True)

    if final_candidate:
        # add a special marker for the solution.
        plt.scatter(solution[0], solution[1], color='red', marker='x', label='Final solution', s=100)
        # add the values of the final candidate to the plot
        plt.text(solution[0], solution[1], f'({round(solution[0],6)}, {round(solution[1],6)})', fontsize=7, va='bottom')
    plt.legend()  # show the legend
    plt.show()

def plot_feasible_region_qp(path, final_candidate=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightpink', alpha=0.5)

    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
    if final_candidate:
        ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='red', marker='x', label='Final candidate')
        ax.text(path[-1][0], path[-1][1], path[-1][2], f'({round(path[-1][0],6)}, {round(path[-1][1],6)}, {round(path[-1][2],6)})', fontsize=7, va='bottom')

    ax.set_title('feasible region')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    ax.view_init(60, 60)
    plt.show()

def plot_final_candidate(solution, objective, constraints):
    fig, ax = plt.subplots()

    # Add the solution, objective, and constraints to the plot
    ax.scatter(solution[0], solution[1], color='red', marker='x', s=100)
    ax.text(solution[0], solution[1],
            f'  x*({solution[0]:.2f}, {solution[1]:.2f})\n'
            f'  obj*({objective:.2f})\n'
            f'  constraints*({", ".join([f"{val:.2f}" for val in constraints])})')

    # Customize the plot
    ax.set_title('Final candidate and its objective and constraint values')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

    # Show the plot
    plt.show()




