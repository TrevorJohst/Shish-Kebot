import numpy as np
from scipy.spatial import KDTree

# Drake dependencies
from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    PiecewisePolynomial,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    BsplineTrajectory,
    Context,
    PointCloud,
    DiagramBuilder,
    Parser,
    ModelInstanceIndex,
    AddMultibodyPlantSceneGraph,
    Solve,
    Concatenate
)

def RemovePlanarSurface(point_cloud: PointCloud, 
                        tolerance: float = 1e-3, 
                        max_iterations: int = 500
                        ) -> PointCloud:
    """
    Args:
        point_cloud (PointCloud): The point cloud object with a planar surface to remove
        tolerance (float): RANSAC tolerance for determining inliers
        max_iterations (int): Maximum number of iterations for RANSAC

    Returns:
        updated_cloud (PointCloud): The updated point cloud with a planar surface removed
    """
    def fitPlane(points):
        """
        Fits a plane to a set of points and returns the equation for the best fit
        Args:
            points (np.ndarray): Set of points of shape (3, N)
        """
        center = np.mean(points, axis=1)
        cxyzs = points.T - center
        U, S, V = np.linalg.svd(cxyzs)
        normal = V[-1]
        d = -center.dot(normal)
        plane_equation = np.hstack([normal, d])
        return plane_equation

    points = point_cloud.xyzs()

    # Array of ones for calculating distances
    ones = np.ones((1, points.shape[1]))
    stacked_points = np.vstack((points, ones))

    # RANSAC loop
    best_ic = 0
    best_model = np.ones(4)
    for _ in range(max_iterations):
        # Randomly select 3 points
        selected = np.rng.choice(points.shape[1], size=3, replace=False)
        selected_points = point_cloud[:, selected]

        # Fit a plane to the points
        plane = fitPlane(selected_points).reshape(4, 1)

        # Calculate distance from every point to the plane
        distances = np.abs(plane.T @ stacked_points) / np.linalg.norm(plane[:3])

        # Count the number of inliers
        ic = np.sum(distances < tolerance)

        # Store current best model
        if ic > best_ic:
            best_ic = ic
            best_model = plane.flatten()

    # Remove the plane from the point cloud
    plane = best_model.reshape(4, 1)
    distances = (np.abs(plane.T @ stacked_points) / np.linalg.norm(plane[:3])).flatten()
    point_cloud.xyz = points[:, distances >= tolerance]

    return point_cloud

def ProcessPointCloud(cameras: list[ModelInstanceIndex],
                      plant: MultibodyPlant,
                      plant_context: Context,
                      crop_lower: tuple[float, float, float] = None,
                      crop_upper: tuple[float, float, float] = None,
                      remove_plane: bool = False
                      ) -> PointCloud:
    """
    Extracts point clouds from a number of cameras and does basic processing on them.
    Args:
        cameras ([ModelInstanceIndex]): The model instances of each camera in the plant
        plant (MultibodyPlant): Plant instance containing the cameras
        plant_context (Context): Plant context to evaluate camera observations at
        crop_lower ((float, float, float)): Lower bounds of the crop region (x, y, z)
        crop_upper ((float, float, float)): Upper bounds of the crop region (x, y, z)
        remove_plane (bool): If True, attemps to remove a planar surface from each cloud
    """
    if (crop_lower and not crop_upper) or (crop_upper and not crop_lower):
        raise RuntimeError("Must pass in both crop_upper and crop_lower if one is passed.")

    pcd = []
    for i, camera in enumerate(cameras):
        # Get the pointcloud for this camera
        # TODO: Check if this works, state output might not exist for a camera
        cloud = plant.get_state_output_port(camera).Eval(plant_context)

        # Crop to region of interest
        if crop_lower:
            pcd.append(cloud.Crop(lower_xyz=crop_lower, upper_xyz=crop_upper))

        # Remove a planar surface
        if remove_plane:
            pcd[i] = RemovePlanarSurface(pcd[i])

        # Estimate normals
        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        # Direct normals from observation direction
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds
    merged_pcd = Concatenate(pcd)

    # Voxelize down-sample
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

    return down_sampled_pcd