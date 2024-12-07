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
)

from ShishKebot.Planning import CreateTrajectoryOptimized


class TrajectoryPublisher(LeafSystem):
    def __init__(
        self, plant, iiwa_name: str = ("iiwa",), end_effector_name: str = "wsg"
    ):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.start_time = self.plant_context.get_time()
        self.iiwa = plant.GetModelInstanceByName(iiwa_name)
        self.wsg = plant.GetModelInstanceByName(end_effector_name)
        self.input = self.DeclareVectorInputPort("target_position", 6)
        self.q_in = self.DeclareVectorInputPort("iiwa_position_measured", 7)
        self.DeclareVectorOutputPort("iiwa_position_cmd", 7, self.SampleTrajectory)

        self.goal = None
        self.trajectory = None

    def generateTrajectory(self):

        current = self.plant.CalcRelativeTransform(
            self.plant_context,
            self.plant.world_frame(),
            self.plant.GetBodyByName("body", self.wsg).body_frame(),
        )
        self.trajectory = CreateTrajectoryOptimized(
            current, self.goal, self.plant, self.plant_context, tol=0.1
        )

    def SampleTrajectory(self, context: Context, output: OutputPort):
        cur_time = context.get_time()
        goal = self.input.Eval(context)
        if goal is not self.goal:
            self.goal = goal
            self.plant.SetPositions(
                self.plant_context, self.iiwa, self.q_in.Eval(context)
            )
            cur_time = context.get_time()
            self.generateTrajectory()
        dt = cur_time - self.start_time
        q = self.trajectory.value(dt)
        output.SetFromVector(q)
