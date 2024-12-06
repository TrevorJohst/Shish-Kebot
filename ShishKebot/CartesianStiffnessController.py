# External libraries
import numpy as np

# Drake dependencies
from pydrake.all import (
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    Context,
    OutputPort,
    RollPitchYaw,
    RotationMatrix,
)


class CartesianStiffnessController(LeafSystem):
    """
    Wrapper system for commanding cartesian stiffness poses to an iiwa.
    Note that the desired position is a 6x1 vector, where the first three
    components are the roll-pitch-yaw orientation and the last 3 components are
    the translational position.

    Input:
        iiwa_position_measured
        iiwa_velocity_measured
        pose_desired

    Output Ports:
        iiwa_torque_cmd
        iiwa_position_cmd
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        iiwa_name: str = "iiwa",
        end_effector_name: str = "wsg",
    ) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName(iiwa_name)
        wsg = plant.GetModelInstanceByName(end_effector_name)
        self._G = plant.GetBodyByName("body", wsg).body_frame()
        self._W = plant.world_frame()
        self._joint_indices = plant.GetActuatedJointIndices(self._iiwa)
        self._joint_indices = [int(i) for i in self._joint_indices]
        self._q0 = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])

        self._debug_count = 0
        self._debug_rate = 6000
        np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

        # Control gains
        self._Kp_pos = 1.0
        self._Kp_rot = 1.0
        self._Kv_pos = 1.0
        self._Kv_rot = 1.0
        self._Kvq = 1.0

        # Controller inputs
        self._q_in = self.DeclareVectorInputPort("iiwa_position_measured", 7)
        self._qdot_in = self.DeclareVectorInputPort("iiwa_velocity_measured", 7)
        self._x_d_in = self.DeclareVectorInputPort("pose_desired", 6)

        # Controller outputs
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 7, self.CalcTorqueOutput)
        self.DeclareVectorOutputPort("iiwa_position_cmd", 7, self.CalcPositionOutput)

    def CalcPositionOutput(self, context: Context, output: OutputPort) -> None:
        """
        Set output position to current position to allow pure torque control
        """
        q = self._q_in.Eval(context)
        output.SetFromVector(q)

    def CalcTorqueOutput(self, context: Context, output: OutputPort) -> None:
        """
        Calculates the joint-space force commands required to achieve a desired
        end effector spatial pose from Katib, Oussama 1987 (doi: 10.1109/JRA.1987.1087068)
        """

        # Update context (why is this required)
        q = self._q_in.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        ###############
        # Initial setup
        ###############

        # Gain setting
        k_vq = self._Kvq  # Scalar
        k_p = np.diag([self._Kp_rot] * 3 + [self._Kp_pos] * 3)  # 6x6
        k_v = np.diag([self._Kv_rot] * 3 + [self._Kv_pos] * 3)  # 6x6

        # External inputs
        qdot = self._qdot_in.Eval(context).reshape(-1, 1)  # 7x1
        x_d = self._x_d_in.Eval(context).reshape(-1, 1)  # 6x1
        xdot_d = np.zeros((6, 1))  # 6x1 CURRENTLY DISABLED

        # Jacobian calculation for current configuration
        J = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,  # Context for function to perform calculations
            JacobianWrtVariable.kQDot,  # Variable to take Jacobian w.r.t
            self._G,  # Frame we want Jacobian of (Gripper)
            [0, 0, 0],  # Specific position in G frame to take Jacobian of
            self._W,  # Frame that qdot is measured in (World)
            self._W,  # Frame we want Jacobian expressed in (World)
        )[:, self._joint_indices]
        # 6x7

        # Measured pose in cartesian space
        X_G = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)
        x = np.vstack((RollPitchYaw(X_G.rotation()).vector(), X_G.translation())).reshape(-1, 1)
        # 6x1         3x1                                    3x1

        # Measured SPATIAL velocity in cartesian space
        xdot = J @ qdot
        # 6x1  6x7 7x1

        ######################
        # Control calculations
        ######################

        # Joint space Coriolis and centrifugal biases (39)
        btilde_r = self._plant.CalcBiasTerm(self._plant_context)[self._joint_indices].reshape(-1, 1)
        # 7x1                  7x1

        # Joint space "mass matrix" (calculated directly internally?)
        A = self._plant.CalcMassMatrix(self._plant_context)[np.ix_(self._joint_indices, self._joint_indices)]
        # 7x7           7x7

        # Joint space forces due to gravity
        # NOTE: This may have to be disabled due to issues with modeling torque-only control
        g = A @ self._plant.CalcGravityGeneralizedForces(self._plant_context)[self._joint_indices].reshape(-1, 1)
        # 7x1 7x7           7x1

        # Pseudo-"mass matrix" for cartesian space (51)
        Lambda_r = np.linalg.inv(J @ np.linalg.inv(A) @ J.T)
        # 6x6                    6x7               7x7  7x6

        # Measured joint-space velocity (64)
        Gamma_s = -k_vq * A @ qdot
        # 7x1      scalar 7x7 7x1

        # Commanded cartesian velocity (66)
        F_rs = k_vq * xdot
        # 6x1  scalar 6x1

        # Desired end effector force (31)
        Fstar_m = -k_p @ self._PoseDifference(x, x_d) - k_v @ (xdot - xdot_d)
        # 6x1      6x6        6x1

        # Calculation of joint force commands (67)
        Gamma = J.T @ Lambda_r @ (Fstar_m + F_rs) + Gamma_s + btilde_r  # + g
        # 7x1 = 7x6   6x6         6x1       6x1     7x1       7x1        7x1

        # Why do I need to manually clip the effort limits?
        output.SetFromVector(np.clip(Gamma, -40, 40))

        self._debug_count += 1
        if self._debug_count > self._debug_rate:
            self._debug_count = 0

    def SetGains(
        self,
        position: tuple[float, float],
        orientation: tuple[float, float],
        null_space: float,
    ) -> None:
        """
        Set the gains for position, orientation, and null-space stability. Tuples are in order
        of (Kp, Kv).
        """
        self._Kp_pos, self._Kv_pos = position
        self._Kp_rot, self._Kv_rot = orientation
        self._Kvq = null_space

    def _PoseDifference(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Computes the 6x1 vector representing the pose error from x1 to x2.
        """

        def vee(S: np.ndarray) -> np.ndarray:
            """vee operator for a 3x3 skew-symmetric matrix"""
            return np.array([-S[1, 2], S[0, 2], -S[0, 1]]).reshape(-1, 1)

        # Extract rotations from RPY representations
        R_1W = RotationMatrix(RollPitchYaw(x1[:3])).matrix()
        R_2W = RotationMatrix(RollPitchYaw(x2[:3])).matrix()

        # Rotation error between x1 and x2 (small angle approximation)
        R_21 = R_2W @ R_1W.T
        rot_error = vee(1 / 2 * (R_21 - R_21.T))

        # Position error between x1 and x2
        pos_error = x1[3:] - x2[3:]

        return np.vstack((rot_error, pos_error)).reshape(-1, 1)

    def _debug(self, print_string):
        if self._debug_count == self._debug_rate:
            print(print_string)
