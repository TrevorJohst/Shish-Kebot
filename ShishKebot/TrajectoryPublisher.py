# External libraries
import numpy as np

# Drake dependencies
from pydrake.all import (
    AbstractValue,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    Context,
    OutputPort,
    RollPitchYaw,
    RotationMatrix,
    RigidTransform,
    PiecewisePose
)

from ShishKebot.Planning import CreateTrajectoryOptimized


class TrajectoryPublisher(LeafSystem):
    """
    Calculates and commands a trajectory either in joint space
    or end effector pose space. Joint space trajectories are optimized,
    while pose space trajectories are simple linear interpolations.

    Input Ports:
        iiwa_position_measured
        pose_desired

    Output Ports:
        iiwa_position_cmd OR iiwa_pose_cmd
    """

    def __init__(self, 
                 plant, 
                 iiwa_name: str = "iiwa", 
                 end_effector_name: str = "wsg",
                 pose_speed: float = None
                 ) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._start_time = self._plant_context.get_time()

        self._iiwa = plant.GetModelInstanceByName(iiwa_name)
        wsg = plant.GetModelInstanceByName(end_effector_name)
        self._G = plant.GetBodyByName("body", wsg).body_frame()
        self._W = plant.world_frame()

        self.joint_space = pose_speed is None
        self.pose_speed = pose_speed

        # System inputs
        self._x_d_in = self.DeclareAbstractInputPort(
            "pose_desired", 
            AbstractValue.Make(RigidTransform())
        )
        self._q_in = self.DeclareVectorInputPort("iiwa_position_measured", 7)

        # System outputs
        if self.joint_space:
            self.DeclareVectorOutputPort(
                "iiwa_position_cmd", 
                7, 
                self.SampleTrajectory
            )
        else:
            self.DeclareAbstractOutputPort(
                "iiwa_pose_cmd", 
                lambda: AbstractValue.Make(RigidTransform()), 
                self.SampleTrajectory
            )

        self.X_goal = None
        self.trajectory = None

    def SampleTrajectory(self, context: Context, output: OutputPort):
        """
        Callback on output function, samples trajectory at current delta time.
        """
        cur_time = context.get_time()

        X_goal = self._x_d_in.Eval(context)
        if not X_goal.IsExactlyEqualTo(self.X_goal):
            # Update plant positions
            self._plant.SetPositions(self._plant_context, self._iiwa, self._q_in.Eval(context))

            # Update trajectory and sampling
            self._start_time = cur_time
            self.X_goal = X_goal
            self._GenerateTrajectory()

        dt = cur_time - self._start_time

        # Sample the trajectory appropriately
        if self.joint_space:
            output.SetFromVector(self.trajectory.value(dt))
        else:
            output.set_value(self.trajectory.GetPose(dt))
        
    def _GenerateTrajectory(self):
        """
        Calculates the trajectory between the current position and the goal
        """
        # Current end effector pose
        X_WG = self._plant.CalcRelativeTransform(
            self._plant_context,
            self._W,
            self._G
        )

        # Optimize a joint space trajectory
        if self.joint_space:
            self.trajectory = CreateTrajectoryOptimized(
                X_WG, self.X_goal, self._plant, self._plant_context, tol=0.1
            )

        # Create a linear pose trajectory
        else:
            traj = PiecewisePose()
            travel_time = (self.X_goal.translation() - X_WG.translation()) / self.pose_speed
            self.trajectory = traj.MakeLinear(
                [0, travel_time], 
                [X_WG, self.X_goal]
            )
