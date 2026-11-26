-- One layer of a layered WarpTracer track; see warptracer_bridge's layer_mapper.
-- Each layer has its own map frame, and layer_mapper places it relative to odom.
include "roboracer_cartographer.lua"

options.map_frame = "layer_1_map"
options.publish_to_tf = false
options.publish_tracked_pose = true

-- On ramps the level LiDAR shoots over the walls climbing and hits the road
-- descending, so lean on odometry: 100x the default prior weights (10, 40).
-- On the bridge this took layer 1 from 36% to 94% of occupied cells on a true wall.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 1000
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 4000

-- Beams with no return (over the walls on a slope) arrive at range_max and clear
-- this much free space; 3 m filled the deck without clearing walls (5 m leaked).
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3

return options
