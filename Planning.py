import numpy as np

# Drake dependencies
from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    PiecewisePolynomial,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    PositionConstraint,
    BsplineTrajectory,
    Context,
    OrientationConstraint,
    Solve,
)

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
    q_goal = SolveIK(X_WGoal, plant)
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

    result = Solve(prog)
    assert result.is_success()
    return result.GetSolution(q_variables)   