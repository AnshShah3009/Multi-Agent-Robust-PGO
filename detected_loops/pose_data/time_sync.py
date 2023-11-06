#!/usr/bin/env python

import rospy
import csv
import os
import numpy as np
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs import point_cloud2 as pc2

Odom_Path = "/home/baymax/Desktop/detected_loops/pose_data/sync/odom/4.csv"
GT_Path = "/home/baymax/Desktop/detected_loops/pose_data/sync/gt/4.csv"
PCD_Folder = "/home/baymax/Desktop/detected_loops/pose_data/sync/pcd/4"


class TimeSyncNode:
    def __init__(self):
        rospy.init_node("time_sync_node", anonymous=True)

        # Set up subscribers for the three topics
        gt_sub = Subscriber("/ground_truth", Odometry)
        odom_sub = Subscriber("/odometry/filtered", Odometry)
        pcl_sub = Subscriber("/points", PointCloud2)

        # ApproximateTimeSynchronizer with a slop of 0.1 seconds
        ts = ApproximateTimeSynchronizer(
            [gt_sub, odom_sub, pcl_sub], queue_size=1000, slop=0.005
        )
        ts.registerCallback(self.synced_callback)

        # Initialize variables to store pose data and point clouds
        self.sequence_number = 1
        self.gt_pose_data = []
        self.odom_pose_data = []
        self.point_clouds = []

        rospy.spin()

    def synced_callback(self, gt_msg, odom_msg, pcl_msg):
        # Callback function for synchronized messages
        rospy.loginfo("Received synchronized messages!")

        # Process the synchronized messages as needed
        gt_pose_data = self.process_odom(gt_msg, GT_Path)
        self.gt_pose_data.append(gt_pose_data)

        odom_pose_data = self.process_odom(odom_msg, Odom_Path)
        self.odom_pose_data.append(odom_pose_data)

        point_cloud = self.process_point_cloud(
            pcl_msg, PCD_Folder, self.sequence_number
        )
        self.point_clouds.append(point_cloud)

        # Increment sequence number for the next set of files
        self.sequence_number += 1

    def process_odom(self, odom_msg, folder_path):
        # Process the odometry message and save pose data to CSV
        pose_data = [
            self.sequence_number,
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ]

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save pose data to CSV file
        csv_file_path = folder_path
        with open(csv_file_path, "a") as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header row if the file is empty
            if csvfile.tell() == 0:
                header = [
                    "Index",
                    "Position_X",
                    "Position_Y",
                    "Position_Z",
                    "Orientation_X",
                    "Orientation_Y",
                    "Orientation_Z",
                    "Orientation_W",
                ]
                csv_writer.writerow(header)
            csv_writer.writerow(pose_data)

        return pose_data

    def process_point_cloud(self, pcl_msg, folder_path, sequence_number):
        # Process the point cloud message and save it as a NumPy array
        pc_data = pc2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
        point_cloud_array = np.array(list(pc_data))

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Create a unique identifier for the NumPy array
        np_identifier = str(sequence_number)

        # Save point cloud array to NumPy file with the identifier
        np_file_path = os.path.join(folder_path, np_identifier + "_pcd.npy")
        np.save(np_file_path, point_cloud_array)

        return point_cloud_array


if __name__ == "__main__":
    try:
        time_sync_node = TimeSyncNode()
    except rospy.ROSInterruptException:
        pass
