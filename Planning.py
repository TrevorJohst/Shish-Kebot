import numpy as np

# Drake dependencies
from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    PiecewisePolynomial,
    Solve,
)

def CreateTrajectory(pose_list: list[RigidTransform], 
                     times: list[float],
                     plant: MultibodyPlant
                     ) -> PiecewisePolynomial:
    """
    Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.

    Args:
        pose_list ([RigidTransform]): List of keyframes X_WG for the trajectory to follow
        times ([float]): Times that the corresponding keyframes should occur at
        plant (MultibodyPlant): Plant containing the iiwa

    Returns:
        q_traj (PiecewisePolynomial): The piecewise polynomial trajectory for the controller to follow
    """
    q_knots = []
    q_nominal = np.array(
        [0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0, 0.0, 0.0]
    )  # nominal joint angles for joint-centering.
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")

    for i, pose in enumerate(pose_list):
        ik = InverseKinematics(plant)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()      # Get MathematicalProgram

        if i == 0:
            prog.SetInitialGuess(q_variables, q_nominal)
        else:
            prog.SetInitialGuess(q_variables, q_knots[-1])

        prog.AddCost(np.dot(q_variables - q_nominal, q_variables - q_nominal))

        p_WG = pose.translation()
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=gripper_frame,
            p_BQ=np.zeros(3),
            p_AQ_lower=p_WG,
            p_AQ_upper=p_WG,
        )
        
        R_WG = pose.rotation()
        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=R_WG,
            frameBbar=gripper_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=0.1,
        )

        result = Solve(prog)
        assert result.is_success()

        q_knots.append(result.GetSolution(q_variables))

    q_knots = np.array(q_knots)

    return PiecewisePolynomial.CubicShapePreserving(times, q_knots[:, 0:7].T)