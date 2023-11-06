import os
import string
import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse

save_path = "/media/baymax/1A52-86C0/new/detected_loops/pose_data/gt_downsampled.pcd"

# csv_files = [
#     f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/1.csv",
#     f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/2.csv",
#     f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/3.csv",
#     f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/4.csv",
# ]

pcd_folders = [
    "/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/pcd/1",
    "/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/pcd/2",
    "/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/pcd/3",
    "/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/pcd/4",
]


class MapBuilder:
    def __init__(self, csv_file_paths, pcd_folder_paths):
        self.global_point_cloud = o3d.geometry.PointCloud()
        self.load_data(csv_file_paths, pcd_folder_paths)

        # Visualize or save the map as needed
        pcd_transformed = self.global_point_cloud.voxel_down_sample(voxel_size=0.1)
        # run the map thriugh dbscan

        o3d.visualization.draw_geometries([pcd_transformed])
        # uncomment to save
        # o3d.io.write_point_cloud(save_path, self.global_point_cloud)
        # o3d.io.write_point_cloud(save_path, pcd_transformed)

    def load_data(self, csv_file_paths, pcd_folder_paths):
        for csv_file, pcd_folder in zip(csv_file_paths, pcd_folder_paths):
            poses = self.load_poses_from_csv(csv_file).sort_values(by=0)

            for index, row in tqdm(poses.iterrows()):
                pose = row.tolist()
                path = str(pose[0]).split(".")[0] + "_pcd.npy"
                pcd_file_path = os.path.join(pcd_folder, path)
                self.add_pcd_to_map(pose, pcd_file_path)

    def load_poses_from_csv(self, csv_file):
        poses = pd.read_csv(csv_file, header=None, delimiter=" ", dtype=str)
        return poses

    def quaternion_to_matrix(self, quaternion):
        r = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        return r.as_matrix()

    def add_pcd_to_map(self, pose, pcd_file_path):
        # Load the saved point cloud
        try:
            pcd = np.load(pcd_file_path)
        except FileNotFoundError:
            print(f"Point cloud file not found: {pcd_file_path}")
            return

        # Extract translation and rotation from the pose
        # translation = np.array([float(pose[8]), float(pose[9]), float(pose[10])])
        # rotation_quaternion = np.array([float(pose[11]), float(pose[12]), float(pose[13]), float(pose[14])])
        translation = np.array([float(pose[1]), float(pose[2]), float(pose[3])])
        rotation_quaternion = np.array(
            [float(pose[4]), float(pose[5]), float(pose[6]), float(pose[7])]
        )
        rotation_matrix = self.quaternion_to_matrix(rotation_quaternion)

        # velodyne transform
        # vel_pos = np.array([0.0812, 0.0, 0.409])
        vel_pos = np.array([0.0812, 0.0, 0.0])
        vel_quat = np.array([0.0, 0.0, 0.0, 1.0])
        vel_rot = self.quaternion_to_matrix(vel_quat)

        translation_final = translation + np.dot(rotation_matrix, vel_pos)
        rotation_final = np.dot(rotation_matrix, vel_rot)
        # print(rotation_final, rotation_matrix)
        # Transform the point cloud to the global coordinate system
        pcd_transformed = o3d.geometry.PointCloud()
        pcd_transformed.points = o3d.utility.Vector3dVector(
            np.dot(rotation_final, pcd[:, :3].T).T + translation_final
        )

        # Aggregate the transformed point cloud into the global point cloud
        self.global_point_cloud += pcd_transformed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add two numbers.")

    # Define two required number arguments
    parser.add_argument("folder", type=str, help="Folder containing csv files")
    select = parser.parse_args().folder
    csv_files = [
        f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/1.csv",
        f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/2.csv",
        f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/3.csv",
        f"/media/baymax/1A52-86C0/new/detected_loops/pose_data/sync/{select}/4.csv",
    ]
    map_builder = MapBuilder(csv_files, pcd_folders)
