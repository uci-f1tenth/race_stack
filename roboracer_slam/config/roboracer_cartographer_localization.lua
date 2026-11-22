include "roboracer_cartographer.lua"

TRAJECTORY_BUILDER.pure_localization_trimmer = {
    max_submaps_to_keep = 3,
}
-- Finding the car on the saved map without placing it anywhere in particular. Until a
-- localization run has been matched to the map, cartographer only looks for it with
-- full-map searches on global_sampling_ratio of new nodes (default 0.003; the first node
-- always gets one), and a match only moves the car when the pose graph is next optimized,
-- every optimize_every_n_nodes nodes (default 90). A stationary car adds a node only every
-- max_time_seconds (default 5 s), so it could sit at the map origin for minutes.
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.5
POSE_GRAPH.global_sampling_ratio = 0.05
POSE_GRAPH.optimize_every_n_nodes = 20

return options
