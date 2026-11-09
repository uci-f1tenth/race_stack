include "map_builder.lua"
include "trajectory_builder.lua"

options = {
    map_builder = MAP_BUILDER,
    trajectory_builder = TRAJECTORY_BUILDER,

    map_frame = "map",
    tracking_frame = "base_link",
    published_frame = "odom",
    odom_frame = "odom",
    provide_odom_frame = false,
    publish_frame_projected_to_2d = true,
    use_pose_extrapolator = true,
    use_odometry = true,
    use_nav_sat = false,
    use_landmarks = false,
    num_laser_scans = 1,
    num_multi_echo_laser_scans = 0,
    num_subdivisions_per_laser_scan = 1,
    num_point_clouds = 0,
    lookup_transform_timeout_sec = 0.2,
    submap_publish_period_sec = 0.3,
    pose_publish_period_sec = 5e-3,
    trajectory_publish_period_sec = 30e-3,
    rangefinder_sampling_ratio = 1.0,
    odometry_sampling_ratio = 1.0,
    fixed_frame_pose_sampling_ratio = 1.0,
    imu_sampling_ratio = 1.0,
    landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- The VESC gyro is already fused into /odom by race_bringup's imu_odom, which
-- fixes odometry heading for both local SLAM and the pose graph. Cartographer's
-- own 2D IMU input would also take the accelerometer as gravity, which hard
-- cornering tilts.
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.min_range = 0.06
TRAJECTORY_BUILDER_2D.max_range = 10.0

-- Default 0.55. In the June 12 mapping run, 243 loop closures disagreed with the
-- current estimate by >2 m (up to 9 m) while matching a submap built moments
-- before -- false matches between look-alike corridors -- and scored a median
-- 57.5% vs ~70% for real ones. 0.65 would have rejected 200 of those 243 while
-- keeping two thirds of the good ones.
POSE_GRAPH.constraint_builder.min_score = 0.65

return options
