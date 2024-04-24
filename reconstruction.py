import os
import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def reconstruct_3d(cache_dir, mesh_dir, focal_length):
    os.makedirs(mesh_dir, exist_ok=True)
    image_list, image_pairs, all_2d_points, all_colors, all_matches, image_size, all_3d_points, cameras, focal_length = load_data(cache_dir, focal_length)

    final_mesh_path = ""
    camera_positions_path = ""

    for index, (i, j) in enumerate(image_pairs):
        idx_2d_1, idx_2d_2, idx_3d = all_matches[index][0], all_matches[index][1], all_matches[index][2]
        points_2d_1, points_2d_2, point_3ds = all_2d_points[i][idx_2d_1].astype('float64'), all_2d_points[j][idx_2d_2].astype('float64'), np.array(all_3d_points[0], dtype=object)[idx_3d]
        intrinsic_matrix = np.array([[focal_length, 0, image_size[i, 0] / 2], [0, focal_length, image_size[i, 1] / 2], [0, 0, 1]])

        E, mask = cv2.findEssentialMat(points_2d_1, points_2d_2, intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=1)
        idx_2d_1, idx_2d_2, idx_3d = idx_2d_1[mask.ravel() == 1], idx_2d_2[mask.ravel() == 1], idx_3d[mask.ravel() == 1]
        points_2d_1, points_2d_2, point_3ds = points_2d_1[mask.ravel() == 1], points_2d_2[mask.ravel() == 1], point_3ds[mask.ravel() == 1]

        mask_ = np.array([pt is None for pt in point_3ds])

        if index != 0:
            ret, rvecs, t, inliers = cv2.solvePnPRansac(np.stack(point_3ds[mask_ == 0]), points_2d_2[mask_ == 0], intrinsic_matrix, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvecs)
        else:
            _, R, t, _ = cv2.recoverPose(E, points_2d_1, points_2d_2, intrinsic_matrix)

        cameras[j] = np.hstack((R, t))

        triangulate(i, j, points_2d_1[mask_ == 1], points_2d_2[mask_ == 1], idx_2d_1[mask_ == 1], idx_2d_2[mask_ == 1], idx_3d[mask_ == 1], intrinsic_matrix, all_3d_points, cameras, all_colors)

    mask = np.array([pt is None for pt in all_3d_points[0]])

    np.save(os.path.join(cache_dir, "camera_poses.npy"), np.array(cameras))
    
    final_mesh_path = os.path.join(mesh_dir, "final_mesh.ply")
    camera_positions_path = os.path.join(mesh_dir, "camera_positions.ply")


    to_ply(os.path.join(mesh_dir, "final_mesh.ply"), np.stack(np.array(all_3d_points[0], dtype=object)[mask == 0]).astype(float),
           np.stack(np.array(all_3d_points[1], dtype=object)[mask == 0]).astype(float))
    to_ply(os.path.join(mesh_dir, "camera_positions.ply"), np.array([(cam[:3, :3].T.dot(np.array([[0, 0, 0]]).T) - cam[:3, 3][:, np.newaxis])[:, 0] for cam in cameras]),
           np.array([np.array([1, 1, 1]) for cam in cameras]) * 255)
    
    return {
        "final_mesh_path": final_mesh_path,
        "camera_positions_path": camera_positions_path
    }

def load_data(cache_dir, focal_length):
    image_list = []
    with open(os.path.join(cache_dir, "visual_list.txt")) as f:
        image_list = [line.strip() for line in f.readlines()]

    image_pairs = np.load(os.path.join(cache_dir, "visual_pairs.npy"), allow_pickle=True)
    all_2d_points = np.load(os.path.join(cache_dir, "extracted_points.npy"), allow_pickle=True)
    all_colors = np.load(os.path.join(cache_dir, "color_data.npy"), allow_pickle=True)
    all_matches = np.load(os.path.join(cache_dir, "point_pairings.npy"), allow_pickle=True)
    image_size = np.load(os.path.join(cache_dir, "extracted_size_data.npy"), allow_pickle=True)

    all_3d_points = [[None] * (np.max(np.hstack(all_matches[:, 2])) + 1), [None] * (np.max(np.hstack(all_matches[:, 2])) + 1)]

    cameras = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]) for _ in range(len(image_list))]

    # focal_length = 2405.05

    return image_list, image_pairs, all_2d_points, all_colors, all_matches, image_size, all_3d_points, cameras, focal_length


def triangulate(i, j, points_2d_1, points_2d_2, idx_2d_1, idx_2d_2, idx_3d, intrinsic_matrix, all_3d_points, cameras, all_colors):
    points_3d = cv2.triangulatePoints(np.matmul(intrinsic_matrix, cameras[i]), np.matmul(intrinsic_matrix, cameras[j]), points_2d_1.T, points_2d_2.T)
    # points_3d = cv2.triangulatePoints(np.matmul(K, cameras[i]), np.matmul(K, cameras[j]), pts0.T, pts1.T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]

    for w, f in enumerate(idx_3d):
        all_3d_points[0][f] = points_3d[w]
        all_3d_points[1][f] = all_colors[i][idx_2d_1[w]]

    x = np.hstack((cv2.Rodrigues(cameras[j][:3, :3])[0].ravel(), cameras[j][:3, 3].ravel(), np.stack(np.array(all_3d_points[0], dtype=object)[idx_3d]).ravel()))
    A = ba_sparse(idx_3d, x)
    res = least_squares(calculate_reprojection_error, x, jac_sparsity=A, x_scale='jac', ftol=1e-8, args=(intrinsic_matrix, points_2d_2))
    R, t, point_3D = cv2.Rodrigues(res.x[:3])[0], res.x[3:6], res.x[6:].reshape(len(idx_3d), 3)

    for w, f in enumerate(idx_3d):
        all_3d_points[0][f] = point_3D[w]

    cameras[j] = np.hstack((R, t.reshape((3, 1))))


def to_ply(file_path, point_cloud, colors):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    shape_info = (out_colors.shape, out_points.shape)
    print(shape_info)
    verts = np.hstack([out_points, out_colors])
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    with open(file_path, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

    return shape_info


def ba_sparse(point3d_indices, x):
    A = lil_matrix((len(point3d_indices) * 2, len(x)), dtype=int)
    A[np.arange(len(point3d_indices) * 2), :6] = 1
    for i in range(3):
        A[np.arange(len(point3d_indices)) * 2, 6 + np.arange(len(point3d_indices)) * 3 + i] = 1
        A[np.arange(len(point3d_indices)) * 2 + 1, 6 + np.arange(len(point3d_indices)) * 3 + i] = 1
    return A


def calculate_reprojection_error(x, intrinsic_matrix, point_2D):
    R, t, point_3D = x[:3], x[3:6], x[6:].reshape((len(point_2D), 3))
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, intrinsic_matrix, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    return (point_2D - reprojected_point).ravel()


if __name__ == "__main__":
    cache_dir = "./dataset03/cache"  # Specify the manual cache directory
    mesh_dir = "./dataset03/meshes"  # Specify the manual mesh directory
    reconstruct_3d(cache_dir, mesh_dir, 2382.05)
