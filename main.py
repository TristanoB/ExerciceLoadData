import torch
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random


class NetCDFDataset(Dataset):
    def __init__(self, file_paths):
    
        self.file_paths = file_paths
        
        

    def __len__(self):
        return len(self.file_paths)

   
    
    def __getitem__(self, idx):
    
        file_path = self.file_paths[idx]
        with nc.Dataset(file_path, 'r') as dataset:
            
            data_x = dataset['x_c'][:]
            data_y = dataset['y_c'][:]
            data_z = dataset['z_c'][:]
            data_t = dataset['t'][:]
            data = dataset['__xarray_dataarray_variable__'][:]
            
            tensor_data = torch.tensor(data, dtype=torch.float32)
        return tensor_data

def print_channels_fichier(file_path) :
    with nc.Dataset(file_path, 'r') as dataset:
        print("Liste des variables dans le fichier :")
        for variable in dataset.variables.values():
            print(variable)



def create_dataloader(nc_dir, batch_size, max_files, shuffle=True, num_workers=0):
   
    file_paths = []
    list_files = os.listdir(nc_dir)
    random.shuffle(list_files)
    for file_name in list_files:
        if file_name.endswith('.nc'):
            file_paths.append(os.path.join(nc_dir, file_name))
            if len(file_paths) >= max_files:  # Stop dès qu'on atteint `max_files`
                break
    
    dataset = NetCDFDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def create_gif(data, save_path="animation.gif", interval=200):
    t, z, y, x = data.shape

   
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
    
   
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    
    ax.set_title("3D Data Animation Over Time")

    gradients_magnitude = []
    centroids = []
    densités = []
    scatter = ax.scatter([], [], [], c=[], cmap="viridis", s=50)

    def update(frame):
       
        z_points, y_points, x_points = np.meshgrid(
            np.arange(z), np.arange(y), np.arange(x), indexing="ij"
        )
        x_points = x_points.flatten()
        y_points = y_points.flatten()
        z_points = z_points.flatten()
        colors = data[frame].flatten()

        total_value = np.sum(data[frame])
        centroid_x = np.sum(x_points * data[frame]) / total_value
        centroid_y = np.sum(y_points * data[frame]) / total_value
        centroid_z = np.sum(z_points * data[frame]) / total_value
        centroids.append([centroid_x, centroid_y, centroid_z])
        gradients = np.gradient(data[frame])
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        average_gradient = np.mean(gradient_magnitude)
        gradients_magnitude.append(average_gradient)
        densités.append(np.mean(data[frame]))

        scatter._offsets3d = (z_points, y_points, x_points)
        scatter.set_array(colors)  
        return scatter

    ani = FuncAnimation(
        fig, update, frames=t, blit=True, interval=interval, repeat=True
    )

    
    ani.save(save_path, writer="pillow")
    print(f"Animation sauvegardée : {save_path}")
    plt.close(fig)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

   
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=range(len(centroids)), cmap='viridis', s=50)

    
    ax.set_title("Trajectoire des Centroïdes au Cours du Temps")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=120) 

    # Sauvegarder la figure
    plt.savefig(os.path.join(save_path,"centroids_3D_plot.png"), dpi=300)
    plt.show()
    return densités, gradients_magnitude, centroids

if __name__ == "__main__":
    
    file_path = os.path.join('C:\\Users\\trist\\Documents\\dev\\Exercice_stage', 'parameters.json')
    with open(file_path, 'r') as file:
        parameters = json.load(file)

    nc_directory = "C:\\Users\\trist\\Documents\\dev\\Exercice_stage\\data\\data_soce"
    
    file_paths = [os.path.join(nc_directory, f) for f in os.listdir(nc_directory) if f.endswith('.nc')]
    print(file_path)
    for file in file_paths : 
        print(f'file_path : {file}, channels : {print_channels_fichier(file)}, end')

    variable_name = "temperature"
    
    batch_size = parameters['batch_size']
    max_files = parameters["max_files"]

    dataloader = create_dataloader(nc_directory, batch_size,max_files)


    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch_data.shape}")

        create_gif(batch_data[0], save_path=nc_directory, interval=200)

