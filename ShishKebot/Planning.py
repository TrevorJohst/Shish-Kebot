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
    AddMultibodyPlantSceneGraph,
    Solve,
)

from manipulation.utils import ConfigureParser

def CandidateGrasp(pcd: PointCloud,
                   max_iter: int = 20
                   ) -> RigidTransform:
    """
    Compute and returns a candidate grasp pose on a pointcloud
    Args:
        pcd (PointCloud): Pointcloud of the object
        max_iter (int): Maximum iterations to allow before aborting search
    Returns:
        X_WG (RigidTransform): Pose of the candidate grasp, None if not found
    """
    # Create a test scene with a WSG
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
    plant.Finalize()

    # Build scene and get contexts
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    
    def SDF(pcd, X_G):
        """
        Compute and return the signed distance for a query pose
        Args:
            pcd (PointCloud): Point cloud to compute distance on
            X_G (RigidTransform): Pose of the gripper to query
        """
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G)
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

        # Compute the signed distance
        pcd_sdf = np.inf
        for pt in pcd.xyzs().T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for body_index in range(len(distances)):
                distance = distances[body_index].distance
                if distance < pcd_sdf:
                    pcd_sdf = distance

        return pcd_sdf

    def darbouxFrame(index, pcd, kdtree, ball_radius=0.002, max_nn=50):
        """
        Given a index of the pointcloud, return a RigidTransform from world to the
        Darboux frame at that point.
        Args:
            index (int): index of the pointcloud.
            pcd (PointCloud): pointcloud of the object.
            kdtree (KDTree): kd tree to use for nn search.
            ball_radius (float): ball_radius used for nearest-neighbors search
            max_nn (int): maximum number of points considered in nearest-neighbors search.
        """
        points = pcd.xyzs()     # 3xN np array of points
        normals = pcd.normals() # 3xN np array of normals

        neighbors = PointCloud(max_nn)

        # Find distances to neighbors at query index
        query = pcd.xyz(index)
        distances, indices = kdtree.query(
                query, k=max_nn, distance_upper_bound=ball_radius
            )

        indices = indices[indices != normals.shape[1]]
        distances = distances[distances != np.inf]
        neighbors.resize(len(distances))
        neighbors.mutable_xyzs()[:] = points[:, indices ]

        # Calculate the N matrix
        N = normals[:, indices] @ normals[:, indices].T

        # Generate the rotation matrix for the normal
        eigenvalues, eigenvectors = np.linalg.eig(N)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]

        # Get basis vectors
        v1 = eigenvectors[:, 2]
        if np.dot(v1, normals[:, index]) > 0: v1 = -v1
        v2 = eigenvectors[:, 1]
        v3 = eigenvectors[:, 0]

        R = np.column_stack((v2, v1, v3))

        if np.linalg.det(R) < 0:
            R = np.column_stack((-v2, v1, v3))

        return RigidTransform(RotationMatrix(R), query)

    def minDistance(pcd, X_WG):
        """
        By doing line search, computes the maximum allowable distance along the y axis before penetration.
        Return the maximum distance, as well as the new transform.
        Args:
            pcd (PointCloud): Pointcloud of the object
            X_WG (RigidTransform): Gripper pose being investigated
        Returns:
            (signed_distance, X_WGnew) ((float, RigidTransform)):
                - signed_distance: Signed distance between gripper and object pointcloud at X_WGnew
                - X_WGnew: New rigid transform that moves X_WG along the y axis while maximizing the 
                  y-translation subject to no collision. 

            If there is no value of y that results in no collisions, returns (np.nan, None)
        """
        y_grid = np.linspace(-0.05, 0.05, 10)

        # Test potential transforms while moving gripper pose around
        signed_distance = np.nan
        X_WGnew = None
        for point in y_grid:
            X_WGtest = RigidTransform(X_WG.rotation(), X_WG.translation() + X_WG.rotation() @ np.array([0, point, 0]))

            # Update pose if not in collision
            distance = SDF(pcd, X_WGtest)
            if distance > 0:
                X_WGnew = X_WGtest
                signed_distance = distance

        return signed_distance, X_WGnew

    def notEmpty(pcd, X_WG):
        """
        Check if the "closing region" of the gripper is nonempty by transforming the pointclouds to gripper coordinates.
        Returns True if the region is not empty, and False if it is.
        Args:
            pcd (PointCloud): pointcloud of the object
            X_WG (RigidTransform): transform of the gripper
        """
        pcd_W_np = pcd.xyzs()

        # Bounding box of the closing region written in the coordinate frame of the gripper body
        crop_min = [-0.05, 0.1, -0.00625]
        crop_max = [0.05, 0.1125, 0.00625]

        # Transform the pointcloud to gripper frame
        X_GW = X_WG.inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)

        # Check if there are any points within the cropped region
        indices = np.all(
            (
                crop_min[0] <= pcd_G_np[0, :],
                pcd_G_np[0, :] <= crop_max[0],
                crop_min[1] <= pcd_G_np[1, :],
                pcd_G_np[1, :] <= crop_max[1],
                crop_min[2] <= pcd_G_np[2, :],
                pcd_G_np[2, :] <= crop_max[2],
            ),
            axis=0,
        )

        return indices.any()

    # Grasp parameters
    x_min = -0.03
    x_max = 0.03
    phi_min = -np.pi / 3
    phi_max = np.pi / 3

    # Build KD tree for the pointcloud.
    kdtree = KDTree(pcd.xyzs().T)
    ball_radius = 0.002
    
    # Try to grasp random points on the cloud until one succeeds
    best = None
    for _ in range(max_iter):
        rand_index = np.random.randint(0, pcd.xyzs().shape[1])

        X_WF = darbouxFrame(rand_index, pcd, kdtree, ball_radius)

        x = np.random.uniform(x_min, x_max)
        phi = np.random.uniform(phi_min, phi_max)
        X_FT = RigidTransform(RotationMatrix.MakeZRotation(phi), np.array([x, 0, 0]))

        dist, X_WG = minDistance(pcd, X_WF @ X_FT)
        if np.isnan(dist): continue

        if notEmpty(pcd, X_WG):
            return X_WG
        
        if not best: best = X_WG
    
    return best

def CreateTrajectoryOptimized(X_WStart: RigidTransform,
                              X_WGoal: RigidTransform,
                              plant: MultibodyPlant,
                              plant_context: Context,
                              max_time: float = 50.0,
                              tol: float = 0.01
                              ) -> BsplineTrajectory:
    """
    Creates a trajectory between two end effector poses using trajectory optimization.
    Args:
        X_WStart (RigidTransform): Pose of the end effector at the start
        X_WGoal (RigidTransform): Pose of the end effector at the goal
        plant (MultibodyPlant): Plant containing the iiwa
        plant_context (Context): Context of the plant from root, used for collision avoidance
        max_time (float): Maximum time to accept for a trajectory
        tol (float): Tolerance for aligning with start and end pose
    Returns:
        q_traj (BsplineTrajectory): The trajectory object generated
    """
    num_q = plant.num_positions()
    tols = np.ones(num_q) * tol
    q0 = plant.GetPositions(plant_context)
    gripper_frame = plant.GetFrameByName("body")

    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 10)
    prog = trajopt.get_mutable_prog()

    # Initial guess
    q_guess = np.tile(q0.reshape((7, 1)), (1, trajopt.num_control_points()))
    q_guess[0, :] = np.linspace(0, -np.pi / 2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    # Default constraints
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
    trajopt.AddDurationConstraint(0.5, max_time)

    # Start constraint
    start_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WStart.translation(),
        X_WStart.translation(),
        gripper_frame,
        [0, 0.1, 0],
        plant_context,
    )
    trajopt.AddPathPositionConstraint(start_constraint, 0)

    # Goal constraint
    q_goal = SolveIK(X_WGoal, plant, pos_tol=tol, rot_tol=tol)
    trajopt.AddPathPositionConstraint(lb=q_goal-tols, ub=q_goal+tols, s=1)

    # Start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

    # # Collision constraints
    # collision_constraint = MinimumDistanceLowerBoundConstraint(
    #     plant, 0.001, plant_context, None, 0.01
    # )
    # evaluate_at_s = np.linspace(0, 1, 50)
    # for s in evaluate_at_s:
    #     trajopt.AddPathPositionConstraint(collision_constraint, s)

    result = Solve(prog)
    assert result.is_success()

    return trajopt.ReconstructTrajectory(result)

def CreateTrajectoryKeyframe(pose_list: list[RigidTransform], 
                             times: list[float],
                             plant: MultibodyPlant,
                             plant_context: Context,
                             pos_tol: float = 0.01,
                             rot_tol: float = 0.01
                             ) -> PiecewisePolynomial:
    """
    Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.
    Args:
        pose_list ([RigidTransform]): List of keyframes X_WG for the trajectory to follow
        times ([float]): Times that the corresponding keyframes should occur at
        plant (MultibodyPlant): Plant containing the iiwa
        plant_context (Context): Context of the plant from root, used for collision avoidance
        pos_tol (float): Tolerance of following keyframe positions (in m)
        rot_tol (float): Tolerance of following keyframe orientations (in rad)
    Returns:
        q_traj (PiecewisePolynomial): The piecewise polynomial trajectory for the controller to follow
    """
    q_knots = []

    for pose in pose_list:
        q_knots.append(SolveIK(pose, plant, plant_context, pos_tol, rot_tol))

    q_knots = np.array(q_knots)

    return PiecewisePolynomial.CubicShapePreserving(times, q_knots[:, 0:7].T)

def SolveIK(pose: RigidTransform,
            plant: MultibodyPlant,
            plant_context: Context = None,
            pos_tol: float = 0.01,
            rot_tol: float = 0.01
            ) -> np.ndarray:
    """
    Solves for the joint positions at a given end effector pose.
    Args:
        pose (RigidTransform): X_WG to solve IK at
        plant (MultibodyPlant): Plant containing the iiwa
        plant_context (Context): Context of the plant from root, avoids collisions if passed in
        pos_tol (float): Tolerance of positions (in m)
        rot_tol (float): Tolerance of orientations (in rad)
    Returns:
        q (np.ndarray): The joint positions at the queried pose
    """
    q_nominal = np.array(
        [0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0]
    )  # nominal joint angles for joint-centering.
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")

    if plant_context: 
        ik = InverseKinematics(plant, plant_context)

        # Collision avoidance constraint
        ik.AddMinimumDistanceLowerBoundConstraint(0.01)
    else:             
        ik = InverseKinematics(plant)

    q_variables = ik.q()  # Get variables for MathematicalProgram
    prog = ik.prog()      # Get MathematicalProgram

    # Joint centering
    prog.AddCost(np.dot(q_variables - q_nominal, q_variables - q_nominal))

    # Constrain position of the gripper
    p_WG = pose.translation()
    tol = np.array([pos_tol, pos_tol, pos_tol])
    ik.AddPositionConstraint(
        frameA=world_frame,
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        p_AQ_lower=p_WG - tol,
        p_AQ_upper=p_WG + tol,
    )
    
    # Constrain orientation of the gripper
    R_WG = pose.rotation()
    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=R_WG,
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=rot_tol,
    )

    lower = plant.GetPositionLowerLimits()
    lower[lower == -np.inf] = -np.pi
    upper = plant.GetPositionUpperLimits()
    upper[upper == np.inf] = np.pi

    for _ in range(100):
        q_rand = np.zeros(len(q_variables))
        q_rand = np.random.uniform(low=lower, high=upper)
        prog.SetInitialGuess(q_variables, q_rand)

        result = Solve(prog)
        if result.is_success():
            return result.GetSolution(q_variables)
        
    raise ValueError