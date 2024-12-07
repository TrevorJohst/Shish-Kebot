from enum import Enum
import numpy as np

from pydrake.all import (
    LeafSystem,
    InputPortIndex,
    AbstractValue,
    RigidTransform,
    MultibodyPlant,
    Context,
    PointCloud,
    RotationMatrix,
    RollPitchYaw
)

from ShishKebot.Perception import ProcessPointCloud
from ShishKebot.Planning import AntipodalCandidateGrasp

class State(Enum):
    START = 1
    PICK_UP_OBJECTS = 2         # diff ik controller to pick up objects
    GET_TO_SKEWER_POSITION = 3  # moves robots to a set positions aligned to skewer
    SKEWER = 4                  # skewering action with stiffness controller
    RELEASE = 5                 # release the food on the skewer

class StateMachine(LeafSystem):
    """
    State machine for switching between states for skewering
    """
    def __init__(
        self,
        iiwa1_plant: MultibodyPlant,
        iiwa2_plant: MultibodyPlant,
        world_plant: MultibodyPlant,
        num_cameras: int,
        iiwa1_name: str = "iiwa1", 
        end_effector1_name: str = "wsg1",
        iiwa2_name: str = "iiwa2", 
        end_effector2_name: str = "wsg2",
        ) -> None:
        LeafSystem.__init__(self)

        # iiwa1 setup
        self._plant1 = iiwa1_plant
        self._plant_context1 = self._plant1.CreateDefaultContext()
        self._iiwa1 = iiwa1_plant.GetModelInstanceByName(iiwa1_name)
        wsg = iiwa1_plant.GetModelInstanceByName(end_effector1_name)
        self._G1 = iiwa1_plant.GetBodyByName("body", wsg).body_frame()

        # iiwa2 setup
        self._plant2 = iiwa2_plant
        self._plant_context2 = self._plant2.CreateDefaultContext()
        self._iiwa2 = iiwa2_plant.GetModelInstanceByName(iiwa2_name)
        wsg = iiwa2_plant.GetModelInstanceByName(end_effector2_name)
        self._G2 = iiwa2_plant.GetBodyByName("body", wsg).body_frame()

        # world setup
        self._world_plant = world_plant
        self._world_context = world_plant.CreateDefaultContext()

        # Predefined state poses
        self.X1_preskewer = RigidTransform(RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 180)), [0.5, 0.5, 0.5]))
        self.X2_preskewer = RigidTransform(RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0.5, -0.5, 0.5]))
        self.X2_skewered = RigidTransform(RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0.5, -0.5, 0.5]))
        self.X2_released = RigidTransform(RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0.5, 0.5, 0.5]))

        # State machine setup
        self.state1 = State.START
        self.state2 = State.START
        self.iiwa1_timer = self._plant_context1.get_time()
        self.iiwa2_timer = self._plant_context2.get_time()

        self.X1_desired = RigidTransform(self.X1_preskewer)
        self.X2_desired = RigidTransform(self.X2_preskewer)
        self.gripper_state = False
        self.num_cameras = num_cameras

        # State machine inputs
        self._q1_in = self.DeclareVectorInputPort("iiwa1_position_measured", 7)
        self._q2_in = self.DeclareVectorInputPort("iiwa2_position_measured", 7)

        # Camera inputs
        for i in range(num_cameras):
            self.DeclareAbstractInputPort(
                f"camera{i}_point_cloud",
                AbstractValue.Make(PointCloud())
            )

        # State machine outputs
        self.DeclareAbstractOutputPort(
            "pose_desired_1", 
            lambda: AbstractValue.Make(RigidTransform()), 
            self.CalcDesiredPose1
        )
        self.DeclareAbstractOutputPort(
            "pose_desired_2", 
            lambda: AbstractValue.Make(RigidTransform()), 
            self.CalcDesiredPose2
        )
        self.DeclareVectorOutputPort("close_gripper", 1, self.CloseGripper)
        # for selecting between stiffness and diff ik controllers
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode
        )

        self._debug_count = 0
        self._debug_rate = 3000
        np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

    def CalcDesiredPose1(self, context, output):

        match self.state1:
            case State.START:
                # Go to preskewer position 
                self.X1_desired = self.X1_preskewer
                self.state1 = State.GET_TO_SKEWER_POSITION

            case State.GET_TO_SKEWER_POSITION:
                # Maintain preskewer position until other iiwa is ready
                if self._AtPose(context, iiwa_num=1, X=self.X1_preskewer) and \
                   self._AtPose(context, iiwa_num=2, X=self.X2_preskewer):
                    # Go to skewering position
                    self.X1_desired = self.X1_skewered
                    self.state1 = State.SKEWER

            case State.SKEWER:
                # Move to skewer objects
                if self._AtPose(context, iiwa_num=1, X=self.X1_skewered):
                    # Go back to preskewer position
                    self.state1 = State.START 

            case _:
                raise RuntimeError("Unexpected state")

        output.set_value(self.X1_desired)

    def CalcDesiredPose2(self, context, output):
        
        match self.state2:
            case State.START:
                if self._AtPose(context, iiwa_num=2, X=self.X2_preskewer):
                    # Take in point clouds and process
                    pcds = []
                    for i in range(self.num_cameras):
                        pcds.append(self.GetInputPort(f"camera{i}_point_cloud").Eval(context))
                    pcd = ProcessPointCloud(
                        pcds, 
                        self._world_plant, 
                        self._world_context,
                        remove_plane=True
                    )

                    # Grasp selection of objects
                    X_desired = AntipodalCandidateGrasp(pcd)

                    if X_desired is None:
                        raise RuntimeError("No grasp found")
                    
                    # Update goal and state
                    self.X2_desired = X_desired
                    self.state2 = State.PICK_UP_OBJECTS

            case State.PICK_UP_OBJECTS:
                # Move to objects
                if self._AtPose(context, iiwa_num=2, X=self.X2_desired):
                    # Grasp objects (close gripper output)
                    self.gripper_state = 0.1

                    # Move to next state
                    self.X2_desired = self.X2_preskewer
                    self.state2 = State.GET_TO_SKEWER_POSITION

                # Timeout after x seconds and retry
                elif self._Timeout(3, iiwa_num=2, context=context):
                    self.X2_desired = self.X2_preskewer
                    self.state2 = State.START

            case State.GET_TO_SKEWER_POSITION:
                # Move to skewer position and maintain
                if self._AtPose(context, iiwa_num=2, X=self.X2_preskewer) and \
                   self._AtPose(context, iiwa_num=1, X=self.X1_skewered):
                    # Release object (open gripper output)
                    self.gripper_state = 1.5

                    # Move to next state
                    self.X2_desired = self.X2_released
                    self.state2 = State.RELEASE

            case State.RELEASE:
                # Move to release the object
                if self._AtPose(context, iiwa_num=1, X=self.X2_released):
                    # Move to next state
                    self.state2 = State.START

            case _:
                raise RuntimeError("Unexpected state")

        output.set_value(self.X2_desired)

        self._debug(self.state2)

        self._debug_count += 1
        if self._debug_count > self._debug_rate:
            self._debug_count = 0

    def CalcControlMode(self, context, output):
        # mode = context.get_abstract_state(int(self._mode_index)).get_value()

        match self.state1:
            case State.SKEWER:
                output.set_value(InputPortIndex(2))  # Use stiffness control
            case _:
                output.set_value(InputPortIndex(1))  # Use diff IK
    
    def CloseGripper(self, context, output):
        output.set_value(self.gripper_state)

    def _AtPose(self, 
                context: Context,
                iiwa_num: int,
                X: RigidTransform,
                tol: float = 0.5
                ) -> bool:
        """
        Evaluates the current pose of the desired iiwa and determines if it is close to X
        """
        if iiwa_num == 1:
            # Update plant
            self._plant1.SetPositions(self._plant_context1, self._iiwa1, self._q1_in.Eval(context))

            # Current pose of the end effector
            X_now = self._plant1.CalcRelativeTransform(
                self._plant_context1,
                self._plant1.world_frame(),
                self._G1
            )

            return X.IsNearlyEqualTo(X_now, tol)

        elif iiwa_num == 2:
            # Update plant
            self._plant2.SetPositions(self._plant_context2, self._iiwa2, self._q2_in.Eval(context))

            # Current pose of the end effector
            X_now = self._plant2.CalcRelativeTransform(
                self._plant_context2,
                self._plant2.world_frame(),
                self._G2
            )

            dist = np.linalg.norm(X.translation() - X_now.translation())
            self._debug(f"Distance: {dist}")
            return dist <= tol

        else:
            raise RuntimeError("iiwa_num was invalid")
        
    def _Timeout(self,
                 duration: float,
                 iiwa_num: int,
                 context: Context
                 ) -> bool:
        """
        Tracks a timeout of specified duration for the specified iiwa
        """
        cur_time = context.get_time()
        if iiwa_num == 1 and cur_time - self.iiwa1_timer > duration:
            print(f"Iiwa{iiwa_num} Timeout")
            self.iiwa1_timer = cur_time
            return True
        elif iiwa_num == 2 and cur_time - self.iiwa2_timer > duration:
            print(f"Iiwa{iiwa_num} Timeout")
            self.iiwa2_timer = cur_time
            return True
        else:
            return False
        
    def _debug(self, print_string):
        if self._debug_count == self._debug_rate:
            print(print_string)