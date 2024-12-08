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

from ShishKebot.Planning import CreateTrajectoryOptimized, CreateTrajectoryRRT, SolveIK


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
                 pose_speed: float = None,
                 tol: float = 0.01
                 ) -> None:
        LeafSystem.__init__(self)
        self._iiwa_name = iiwa_name
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._start_time = self._plant_context.get_time()

        self._iiwa = plant.GetModelInstanceByName(iiwa_name)
        wsg = plant.GetModelInstanceByName(end_effector_name)
        self._G = plant.GetBodyByName("body", wsg).body_frame()
        self._W = plant.world_frame()

        self.joint_space = pose_speed is None
        self.pose_speed = pose_speed
        self.traj_idx = 0
        self.sample_speed = 0.01
        self.tol = tol

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

        self._X_goal_index = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform())
        )
        self.trajectory = None

        self._debug_count = 0
        self._debug_rate = 3000
        np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

    def SampleTrajectory(self, context: Context, output: OutputPort):
        """
        Callback on output function, samples trajectory at current delta time.
        """
        cur_time = context.get_time()
        cur_X_goal = self._X_goal(context)

        X_goal = self._x_d_in.Eval(context)
        if not X_goal.IsExactlyEqualTo(cur_X_goal):
            # Update plant positions
            q_now = self._q_in.Eval(context)
            self._plant.SetPositions(self._plant_context, self._iiwa, q_now)

            # Update trajectory and sampling time (if successful)
            self._set_X_goal(context, X_goal)
            if self._GenerateTrajectory(context):
                self._start_time = cur_time
                self.traj_idx = 0

        dt = cur_time - self._start_time

        # Sample the trajectory appropriately
        if self.joint_space:
            output.SetFromVector(self.trajectory.value(self.traj_idx))
            self.traj_idx += 1*self.sample_speed
        else:
            output.set_value(self.trajectory.GetPose(dt))

        self._debug_count += 1
        if self._debug_count > self._debug_rate:
            self._debug_count = 0
        
    def _GenerateTrajectory(self, context: Context) -> bool:
        """
        Calculates the trajectory between the current position and the goal
        """
        # Current end effector pose
        X_WG = self._plant.CalcRelativeTransform(
            self._plant_context,
            self._W,
            self._G
        )

        # Current goal pose
        X_goal = self._X_goal(context)

        # Optimize a joint space trajectory
        if self.joint_space:
            # This optimization is horrible, RRT never fails me, fuckery to make it work

            # trajectory = None
            # while trajectory is None:
            #     trajectory = CreateTrajectoryOptimized(
            #         X_WG, X_goal, self._plant, self._plant_context, tol=self.tol
            #     )
            # if trajectory is not None:
            #     print(f"{self._iiwa_name} trajectory update")
            #     self.trajectory = trajectory
            #     return True
            # else:
            #     print(f"{self._iiwa_name} failed to update trajectory")
            #     return False

            self.trajectory = CreateTrajectoryRRT(
                X_WG, X_goal, self._plant, self._plant_context, max_iter=1000
            )
            print(f"{self._iiwa_name} trajectory update")
            return True

        # Create a linear pose trajectory
        else:
            traj = PiecewisePose()
            travel_time = (X_goal.translation() - X_WG.translation()) / self.pose_speed
            self.trajectory = traj.MakeLinear(
                [0, travel_time], 
                [X_WG, X_goal]
            )
            print(f"{self._iiwa_name} trajectory update")
            return True

    def _X_goal(self, context: Context):
        return context.get_abstract_state(int(self._X_goal_index)).get_value()
    
    def _set_X_goal(self, context: Context, X: RigidTransform):
        context.get_mutable_abstract_state(int(self._X_goal_index)).set_value(X)

    def _debug(self, print_string):
        if self._debug_count == self._debug_rate:
            print(print_string)