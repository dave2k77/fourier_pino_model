import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os

class HeatEquationSolver:
    def __init__(self, plate_length=150, plate_width=250, max_iter_time=500, alpha=1.0, delta_x=1, delta_y=1):
        self.plate_length = plate_length
        self.plate_width = plate_width
        self.max_iter_time = max_iter_time
        self.alpha = alpha
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_t = (delta_x ** 2) / (4 * alpha)
        self.gamma = (alpha * self.delta_t) / (delta_x ** 2)

        self.u = np.empty((max_iter_time, plate_length, plate_width))
        self.bc_and_ic_setup()

    def bc_and_ic_setup(self, u_initial=0, u_top=100, u_left=0, u_bottom=0, u_right=0):
        self.u.fill(u_initial)
        self.u[:, (self.plate_length - 1):, :] = u_top
        self.u[:, :, :1] = u_left
        self.u[:, :1, 1:] = u_bottom
        self.u[:, :, (self.plate_width - 1):] = u_right

    def evolve(self, save_interval=25): # add argument: filename_prefix="images/heat_equation_evolution" to save
        for k in range(0, self.max_iter_time - 1, 1):
            for i in range(1, self.plate_length - 1, self.delta_x):
                for j in range(1, self.plate_width - 1, self.delta_y):
                    self.u[k + 1, i, j] = self.gamma * (self.u[k][i + 1][j] + self.u[k][i - 1][j] + self.u[k][i][j + 1] + self.u[k][i][j - 1] - 4 * self.u[k][i][j]) + self.u[k][i][j]

            #if (k + 1) % save_interval == 0:
            #    self.save_evolution(f"{filename_prefix}_timestep_{k + 1}.npz", self.u[k + 1])

    
    
    def plotheatmap(self, u_k, k):
        plt.clf()
        plt.title(f"Temperature at t = {k * self.delta_t:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100) # type: ignore
        plt.colorbar()
        return plt

    def animate(self, k):
        return self.plotheatmap(self.u[k], k)
    

    def save_animation(self, filename="movies/heat_equation_solution_alpha10_u50.gif", writer='imagemagick'):
        anim = FuncAnimation(plt.figure(), self.animate, interval=1, frames=self.max_iter_time, repeat=False)
        anim.save(filename, writer=writer)

    def save_evolution(self, filename, u_timestep):
        np.savez(filename, u=u_timestep)


    def save_heatmaps(self):
        for k in range(1, self.max_iter_time-1, 25):
            self.plotheatmap(self.u[k], k).savefig(f"images/png/heatmap_{k * 25}", dpi=300)

    def save_solution_and_image(self, k, output_dir="images/data_u50"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the solution 'u' as a .npz file
        npz_filename = os.path.join(output_dir, f"solution_alpha10_u50_timestep_{k}.npz")
        np.savez(npz_filename, u=self.u[k])

        # Save the corresponding heatmap image as a .png file
        png_filename = os.path.join(output_dir, f"heatmap_alpha10_u50_timestep_{k}.png")
        img = self.plotheatmap(self.u[k], k)
        img.savefig(png_filename, dpi=300)

if __name__ == "__main__":
    print("2D Heat Equation Solver")
    solver = HeatEquationSolver()
    solver.evolve()

    for k in range(0, 500, 25):
        solver.save_solution_and_image(k)
    
    solver.save_animation()
    #solver.save_first_n_images(10)
    #solver.save_heatmaps()