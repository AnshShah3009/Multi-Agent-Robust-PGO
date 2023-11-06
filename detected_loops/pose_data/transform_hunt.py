import rospy
import tf2_ros
import geometry_msgs.msg
import tf
import tf.transformations as tft
import numpy as np
from scipy.spatial.transform import Rotation

def get_transform(tf_buffer, source_frame, target_frame):
    try:
        transform_stamped = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
        return transform_stamped.transform
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn(f"Transform from {source_frame} to {target_frame} not available.")
        return None

def combine_transforms(parent_transform, child_transform):
    # Convert quaternion to rotation matrix for both parent and child
    quaternion = [parent_transform.rotation.x,   parent_transform.rotation.y,  parent_transform.rotation.z,  parent_transform.rotation.w]
    rot_matrix_parent = Rotation.from_quat(quaternion).as_matrix()
    
    quaternion = [child_transform.rotation.x,  child_transform.rotation.y,  child_transform.rotation.z,  child_transform.rotation.w]
    rot_matrix_child = Rotation.from_quat(quaternion).as_matrix()

    # Combine translation
    translation_parent = [parent_transform.translation.x, parent_transform.translation.y, parent_transform.translation.z]
    translation_child = [child_transform.translation.x, child_transform.translation.y, child_transform.translation.z]

    # print(rot_matrix_child)
    combine_translation = np.dot(rot_matrix_parent, translation_child) + translation_parent
    combined_rotation = np.dot(rot_matrix_parent, rot_matrix_child)
    rotation = Rotation.from_matrix(combined_rotation)
    quaternion = rotation.as_quat()

    # Create the combined transform
    transform_combined = geometry_msgs.msg.Transform()
    transform_combined.translation.x = combine_translation[0]
    transform_combined.translation.y = combine_translation[1]
    transform_combined.translation.z = combine_translation[2]
    transform_combined.rotation.x = quaternion[0]
    transform_combined.rotation.y = quaternion[1]
    transform_combined.rotation.z = quaternion[2]
    transform_combined.rotation.w = quaternion[3]

    return transform_combined

def main():
    rospy.init_node('transform_example')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    baselink_frame = 'base_link'
    top_plate_link_frame = 'top_plate_link'
    vlp_16_mount_baselink_frame = 'vlp16_mount_base_link'
    vlp_16_mount_plate_frame = 'vlp16_mount_plate'
    velodyne_baselink_frame = 'velodyne_base_link'
    velodyne_frame = 'velodyne'

    # Get transforms from baselink to velodyne
    transform_baselink_to_top_plate_link = get_transform(tf_buffer, baselink_frame, top_plate_link_frame)
    transform_top_plate_link_to_vlp_16_mount_baselink = get_transform(tf_buffer, top_plate_link_frame, vlp_16_mount_baselink_frame)
    transform_vlp_16_mount_baselink_to_vlp_16_mount_plate = get_transform(tf_buffer, vlp_16_mount_baselink_frame, vlp_16_mount_plate_frame)
    transform_vlp_16_mount_plate_to_velodyne_baselink = get_transform(tf_buffer, vlp_16_mount_plate_frame, velodyne_baselink_frame)
    transform_velodyne_baselink_to_velodyne = get_transform(tf_buffer, velodyne_baselink_frame, velodyne_frame)

    if all(transform is not None for transform in [transform_baselink_to_top_plate_link,
                                                   transform_top_plate_link_to_vlp_16_mount_baselink,
                                                   transform_vlp_16_mount_baselink_to_vlp_16_mount_plate,
                                                   transform_vlp_16_mount_plate_to_velodyne_baselink,
                                                   transform_velodyne_baselink_to_velodyne]):

        # Combine the transforms to get the transform from baselink to velodyne
        transform_baselink_to_velodyne = combine_transforms(transform_baselink_to_top_plate_link,
                                                            transform_top_plate_link_to_vlp_16_mount_baselink)
        transform_baselink_to_velodyne = combine_transforms(transform_baselink_to_velodyne,
                                                            transform_vlp_16_mount_baselink_to_vlp_16_mount_plate)
        transform_baselink_to_velodyne = combine_transforms(transform_baselink_to_velodyne,
                                                            transform_vlp_16_mount_plate_to_velodyne_baselink)
        transform_baselink_to_velodyne = combine_transforms(transform_baselink_to_velodyne,
                                                            transform_velodyne_baselink_to_velodyne)

        print(f"Transform from {baselink_frame} to {velodyne_frame}:")
        print(f"Translation: ({transform_baselink_to_velodyne.translation.x}, {transform_baselink_to_velodyne.translation.y}, {transform_baselink_to_velodyne.translation.z})")
        print(f"Rotation: ({transform_baselink_to_velodyne.rotation.x}, {transform_baselink_to_velodyne.rotation.y}, {transform_baselink_to_velodyne.rotation.z}, {transform_baselink_to_velodyne.rotation.w})")
    else:
        rospy.logwarn("Unable to compute the transform.")

if __name__ == '__main__':
    main()
