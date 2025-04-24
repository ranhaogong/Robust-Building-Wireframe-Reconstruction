import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d
import torch
from torch_cluster import knn
import torch.sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_pts(pts_file, color=False, nir=False, intensity=False):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        data = [line.strip().split(' ') for line in lines]
        
    data = np.array(data, dtype=np.float64)
    if not color and not nir and not intensity:
        return data[:, :3]
    elif color and not nir and not intensity:
        return data[:, :6]
    elif not color and nir and not intensity:
        return data[:, [0, 1, 2, 6]]
    elif not color and not nir and intensity:
        return data[:, [0, 1, 2, 7]]
    elif color and nir and not intensity:
        return data[:, [0, 1, 2, 3, 4, 5, 6]]
    elif color and not nir and intensity:
        return data[:, [0, 1, 2, 3, 4, 5, 7]]
    elif not color and nir and intensity:
        return data[:, [0, 1, 2, 6, 7]]
    else:
        return data[:, :8]

def compute_mrgd_features(points, device='cuda'):
    """Optimized MRGD computation with Gaussian kernels"""
    coord = torch.from_numpy(points[:, :3]).float().to(device)
    N = coord.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # k-NN computation (k=32)
    edge_l = knn(coord, coord, k=32, batch_x=batch, batch_y=batch)
    edge_l = edge_l[:, :N * 32]
    edge_s = edge_l[:, ::4][:, :N * 8]

    # Normals
    n_s = compute_normals_torch(coord, edge_s[1], k=8)
    n_l = compute_normals_torch(coord, edge_l[1], k=32)

    # Curvature
    k_s = compute_curvature(coord, edge_s[1], n_s, k=8)
    k_l = compute_curvature(coord, edge_l[1], n_l, k=32)

    # Height variance
    z = coord[:, 2]
    sigma_s = torch.sqrt(((z[edge_s[1]] - z[edge_s[0]]) ** 2).view(N, 8).mean(dim=1))
    sigma_l = torch.sqrt(((z[edge_l[1]] - z[edge_l[0]]) ** 2).view(N, 32).mean(dim=1))

    # Spectral features
    S_s = compute_graph_spectral_torch(coord, edge_s, k=8, m=3)
    S_l = compute_graph_spectral_torch(coord, edge_l, k=32, m=3)

    # Nonlinear contrast kernels
    sigma = 0.1
    K_c = torch.exp(-((k_s - k_l) ** 2) / (2 * sigma ** 2))
    K_h = torch.exp(-((sigma_s - sigma_l) ** 2) / (2 * sigma ** 2))
    K_s = torch.exp(-((S_s - S_l) ** 2) / (2 * sigma ** 2))

    # MRGD features
    mrgd = torch.stack([K_c * k_s, K_h * sigma_s, K_s * S_s], dim=1)

    # Min-Max normalization
    mrgd_min = mrgd.min(dim=0, keepdim=True)[0]
    mrgd_max = mrgd.max(dim=0, keepdim=True)[0]
    mrgd_normalized = (mrgd - mrgd_min) / (mrgd_max - mrgd_min + 1e-8)

    # Return MRGD and kernels for visualization
    return mrgd_normalized.cpu().numpy(), K_c.cpu().numpy(), K_h.cpu().numpy(), K_s.cpu().numpy(), \
           (k_s - k_l).cpu().numpy(), (sigma_s - sigma_l).cpu().numpy(), (S_s - S_l).cpu().numpy()

def compute_normals_torch(coord, indices, k):
    """Compute normals using PyTorch"""
    N = coord.size(0)
    points = coord[indices].view(N, k, 3)
    centered = points - points.mean(dim=1, keepdim=True)
    cov = torch.bmm(centered.transpose(1, 2), centered) / k
    _, eig_vecs = torch.linalg.eigh(cov)
    normals = eig_vecs[:, :, 0]
    normals = normals * torch.sign((normals * coord).sum(dim=1, keepdim=True))
    return normals

def compute_curvature(coord, indices, normals, k):
    """Simplified curvature computation"""
    N = coord.size(0)
    points = coord[indices].view(N, k, 3)
    vectors = points - coord.unsqueeze(1)
    proj_dist = (vectors * normals.unsqueeze(1)).sum(dim=-1)
    curvature = proj_dist.abs().mean(dim=1)
    return curvature / (curvature.max() + 1e-8)

def compute_graph_spectral_torch(coord, edge_index, k, m=3):
    """GPU-accelerated spectral analysis"""
    N = coord.size(0)
    row, col = edge_index
    values = torch.ones(row.size(0), device=coord.device)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), values, (N, N))
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    D = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N, device=coord.device), torch.arange(N, device=coord.device)]),
        degree, (N, N))
    L = D - adj

    def power_iteration(L, num_iter=10):
        v = torch.randn(N, m, device=coord.device)
        v = v / torch.norm(v, dim=0, keepdim=True)
        for _ in range(num_iter):
            v = L @ v
            v = v / (torch.norm(v, dim=0, keepdim=True) + 1e-8)
        return v

    eigenvectors = power_iteration(L)
    spectral_energy = torch.sum(eigenvectors ** 2, dim=1)
    return spectral_energy / (spectral_energy.max() + 1e-8)

def compute_fpfh_features(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    return np.asarray(fpfh.data).T

def select_best_fpfh_features(all_fpfh_features):
    variances = np.var(all_fpfh_features, axis=0)
    selected_indices = np.argsort(variances)[-5:]
    return selected_indices

def normalize_features(features):
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    return (features - min_val) / (max_val - min_val + 1e-8)

# Visualize Gaussian kernel as 3D surface
def visualize_gaussian_kernel(diff_values, kernel_values, label, output_path):
    # Create a grid for the Gaussian kernel
    x = np.linspace(np.min(diff_values), np.max(diff_values), 100)
    y = x  # Assume symmetric input for visualization
    X, Y = np.meshgrid(x, y)
    sigma = 0.1
    Z = np.exp(-((X**2 + Y**2) / (2 * sigma**2)))  # Gaussian kernel surface

    # Plot 3D surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('Feature Difference (x)')
    # ax.set_ylabel('Feature Difference (y)')
    # ax.set_zlabel('Kernel Value')
    ax.set_title(f'Gaussian Kernel: {label}')
    fig.colorbar(surf, ax=ax, label='Kernel Intensity')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# File paths
input_file = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz_norm/28.xyz"
output_dir = "/data/haoran/Point2Roof/vis_fpfh_mrgd"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Read point cloud
points = read_pts(input_file)
xyz = points[:, :3]
min_pt_xyz = np.min(xyz, axis=0)
max_pt_xyz = np.max(xyz, axis=0)
maxXYZ = np.max(max_pt_xyz)
minXYZ = np.min(min_pt_xyz)
min_pt_xyz[:] = minXYZ
max_pt_xyz[:] = maxXYZ
centroid_xyz = np.mean(xyz, axis=0)
xyz -= centroid_xyz
max_distance_xyz = np.max(np.linalg.norm(xyz, axis=1))
xyz /= max_distance_xyz
points[:, :3] = xyz

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Check point cloud
if points.shape[0] == 0:
    raise ValueError(f"Failed to load point cloud from {input_file}")

# Estimate normals for FPFH
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Compute FPFH and MRGD
all_fpfh_features = compute_fpfh_features(points)
selected_fpfh_indices = select_best_fpfh_features(all_fpfh_features)
fpfh = normalize_features(all_fpfh_features[:, selected_fpfh_indices])
mrgd, K_c, K_h, K_s, diff_curvature, diff_height, diff_spectral = compute_mrgd_features(points)

# Create color mappings for FPFH and MRGD
def create_colors(norm_values, is_fpfh=True):
    colors = np.zeros((len(norm_values), 3))
    norm_values = norm_values / (norm_values.max() + 1e-8)
    if is_fpfh:
        colors[:, 0] = norm_values  # Red
        colors[:, 1] = 1 - norm_values  # Yellow gradient
    else:
        colors[:, 2] = norm_values  # Blue
        colors[:, 1] = 1 - norm_values  # Green gradient
    return colors

# FPFH point cloud
fpfh_norm = np.linalg.norm(fpfh, axis=1)
fpfh_colors = create_colors(fpfh_norm, is_fpfh=True)
pcd_fpfh = o3d.geometry.PointCloud()
pcd_fpfh.points = o3d.utility.Vector3dVector(points)
pcd_fpfh.colors = o3d.utility.Vector3dVector(fpfh_colors)

# MRGD point cloud
mrgd_norm = np.linalg.norm(mrgd, axis=1)
mrgd_colors = create_colors(mrgd_norm, is_fpfh=False)
pcd_mrgd = o3d.geometry.PointCloud()
pcd_mrgd.points = o3d.utility.Vector3dVector(points)
pcd_mrgd.colors = o3d.utility.Vector3dVector(mrgd_colors)

# Save colored point clouds
def save_colored_point_cloud(pcd, output_path):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    data = np.hstack((points, colors))
    np.savetxt(output_path, data, fmt='%.6f %.6f %.6f %.0f %.0f %.0f', header='x y z r g b')

save_colored_point_cloud(pcd_fpfh, os.path.join(output_dir, "fpfh_colored.xyz"))
save_colored_point_cloud(pcd_mrgd, os.path.join(output_dir, "mrgd_colored.xyz"))

# Visualize Gaussian kernels
kernel_labels = ['Curvature', 'Height Variance', 'Spectral Energy']
kernel_values = [K_c, K_h, K_s]
kernel_diffs = [diff_curvature, diff_height, diff_spectral]

for label, K, diff in zip(kernel_labels, kernel_values, kernel_diffs):
    output_path = os.path.join(output_dir, f"kernel_{label.lower().replace(' ', '_')}.png")
    visualize_gaussian_kernel(diff, K, label, output_path)
    print(f"Gaussian kernel visualization saved to {output_path}")

print(f"Colored point clouds saved to {output_dir}/fpfh_colored.xyz and {output_dir}/mrgd_colored.xyz")