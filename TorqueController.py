# External libraries
import numpy as np

# Drake dependencies
from pydrake.all import (
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    Context,
    OutputPort,
    RigidTransform
)

# Type hinting
from typing import Callable


class TorqueController(LeafSystem):
    """
    Wrapper System for Commanding Pure Torques to an iiwa.

    :InputPort(0): q_in
    :InputPort(1): qdot_in
    :OutputPort(0): q_out
    :OutputPort(1): tau_out
    """

    def __init__(self, 
                 plant: MultibodyPlant, 
                 ctrl_fun: Callable[[RigidTransform, np.ndarray, np.ndarray], np.ndarray]
                ) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._ctrl_fun = ctrl_fun

        # Joint centering PD gains
        self.Kp = 0.5
        self.Kd = 0.1

        # Controller inputs
        self._q_in = self.DeclareVectorInputPort("iiwa_position_measured", 7)
        self._qdot_in = self.DeclareVectorInputPort("iiwa_velocity_measured", 7)
        self._xy_des = self.DeclareVectorInputPort("xy_position_desired", 2)

        # Controller outputs 
        self.DeclareVectorOutputPort("iiwa_position_command", 7, self.CalcPositionOutput)
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 7, self.CalcTorqueOutput)

    def CalcPositionOutput(self, context: Context, output: OutputPort) -> None:
        """
        Set q_d = q_now. This ensures the iiwa goes into pure torque mode in sim by 
        setting the position control torques in InverseDynamicsController to zero.
        NOTE(terry-suh): Do not use this method on hardware or deploy this notebook on hardware.
        We can only simulate pure torque control mode for iiwa on sim.
        """
        q_now = self._q_in.Eval(context)
        output.SetFromVector(q_now)

    def CalcTorqueOutput(self, context: Context, output: OutputPort) -> None:

        # Read inputs
        q_now = self._q_in.Eval(context)
        qdot_now = self._qdot_in.Eval(context)
        xy_des = self._xy_des.Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)

        # Convert joint space quantities to Cartesian quantities.
        X_G_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )

        # Only select relevant terms. We get a J_G of shape (6,7)
        # rows: r, p, y, x, y, z
        # cols: q0, q1, q2, q3, q4, q5, q6
        J_G = J_G[:, :7]

        # Calculate spatial velocity in end effector frame
        V_G_now = J_G @ qdot_now

        # Apply control function to get spatial torques
        F_G_cmd = self._ctrl_fun(X_G_now, V_G_now, xy_des)

        # Convert back to joint coordinates
        tau_cmd = J_G.T @ F_G_cmd

        # Joint centering
        q_0 = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])
        tau_cmd = tau_cmd + self._Project(self.Kp * (q_0 - q_now) - self.Kd * (qdot_now), J_G)

        output.SetFromVector(tau_cmd)

    def SetGains(self, Kp: float, Kd: float) -> None:
        self.Kp = Kp
        self.Kd = Kd

    def _Project(self, joint_force_vector: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        Implements the "dynamically-consistent null-space projection" 
        from Katib, Oussama 1987 (doi: 10.1109/JRA.1987.1087068)
        """
        return (np.eye(7) - J.T @ np.linalg.pinv(J).T) @ joint_force_vector