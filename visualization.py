import os
import pyvista as pv
import numpy as np


def visualize_mesh(mesh_dir):
    # Load the PLY file
    ply_filename = os.path.join(mesh_dir, 'final_mesh.ply')
    mesh = pv.read(ply_filename)

    # Set all points to black color
    # black_color = [0, 0, 0]
    # vertex_colors = np.full((mesh.n_points, 3), black_color, dtype=float)
    
    # Generate random vertex colors
    vertex_colors = np.random.random((mesh.n_points, 3))

    # Create a plotter
    plotter = pv.Plotter()

    # Add the mesh to the plotter with specified colors
    mesh['colors'] = vertex_colors  # Add colors as scalar field
    plotter.add_mesh(mesh, scalars='colors', show_edges=True)

    # Adjust camera clipping range
    plotter.camera.clipping_range = [0.01, 1000]  # Adjust the values

    # Create a white light
    white_light = pv.Light(color='white', position=(1, 1, 1), focal_point=(0, 0, 0))

    # Add the light to the plotter
    plotter.add_light(white_light)

    # Show the plotter window
    plotter.show()

if __name__ == "__main__":
    mesh_dir = "./dataset03/meshes" # Provide manual path
    visualize_mesh(mesh_dir)