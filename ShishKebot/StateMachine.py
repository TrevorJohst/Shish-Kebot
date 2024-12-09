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
    RollPitchYaw,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    DiagramBuilder,
    Parser,
    AddMultibodyPlantSceneGraph,
)

from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad

import ShishKebot.Seed
from ShishKebot.Perception import ProcessPointCloud
from ShishKebot.Planning import AntipodalCandidateGrasp

class State(Enum):
    START = 1
    PREGRASP = 2                # above block
    GRASP = 3                   # diff ik controller to pick up objects
    GET_TO_SKEWER_POSITION = 4  # moves robots to a set positions aligned to skewer
    SKEWER = 5                  # skewering action with stiffness controller
    RELEASE = 6                 # release the food on the skewer

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
        meshcat = None
        ) -> None:
        LeafSystem.__init__(self)

        # Visualization setup
        self.meshcat = meshcat
        self.grasp_count = 0

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
        self.X1_preskewer = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, np.pi)), [0.503, 0.45, 0.497])
        # AddMeshcatTriad(meshcat, "X1_preskewer", X_PT=self.X1_preskewer, length=0.15, opacity=0.2)
        self.X2_preskewer = RigidTransform(RotationMatrix(RollPitchYaw(0, np.pi/2, 0)), [0.5, 0.0, 0.54])
        # AddMeshcatTriad(meshcat, "X2_preskewer", X_PT=self.X2_preskewer, length=0.15, opacity=0.2)
        self.X1_skewered = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, np.pi)), [0.503, 0.25, 0.497])
        # AddMeshcatTriad(meshcat, "X1_skewered", X_PT=self.X1_skewered, length=0.15, opacity=0.2)
        self.X2_released = RigidTransform(RotationMatrix(RollPitchYaw(0, np.pi/2, 0)), [0.3, 0.0, 0.5])
        # AddMeshcatTriad(meshcat, "X2_released", X_PT=self.X2_released, length=0.15, opacity=0.2)
        self.X1_post_skewered = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/4, 0, np.pi)), [0.5, 0.3, 0.495])

        # State machine setup
        self._state1_index = self.DeclareAbstractState(
            AbstractValue.Make(State.START)
        )
        self._state2_index = self.DeclareAbstractState(
            AbstractValue.Make(State.START)
        )
        self.iiwa1_timer = self._plant_context1.get_time()
        self.iiwa2_timer = self._plant_context2.get_time()
        self.timer1_reset = False
        self.timer2_reset = False

        self._X1_desired_index = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform(self.X1_post_skewered))
        )
        self._X2_desired_index = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform(self.X2_preskewer))
        )
        self.gripper_open = -80
        self.gripper_closed = 80
        self._gripper_state_index = self.DeclareAbstractState(
            AbstractValue.Make(self.gripper_open)
        )
        self.X_PregraspGrasp = RigidTransform([0, 0, 0.1])
        self.X_Grasp = RigidTransform()
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
        self.DeclareVectorOutputPort("close_gripper", 2, self.CloseGripper)
        # for selecting between stiffness and diff ik controllers
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode
        )

        self._debug_count = 0
        self._debug_rate = 7000
        np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

    def CalcDesiredPose1(self, context, output):
        state = self._GetState(iiwa_num=1, context=context)
        
        match state:
            case State.START:
                if self._AtPose(context, iiwa_num=1, X=self.X1_post_skewered):
                    self._ResetTimeout(context, iiwa_num=1)

                    if self._Timeout(1, iiwa_num=1, context=context):
                        # Go to next state
                        self._X1_desired(context, self.X1_preskewer)
                        self._ChangeState(1, State.GET_TO_SKEWER_POSITION, context)

            case State.GET_TO_SKEWER_POSITION:
                # Maintain preskewer position until other iiwa is ready
                if self._AtPose(context, iiwa_num=1, X=self.X1_preskewer) and \
                   self._AtPose(context, iiwa_num=2, X=self.X2_preskewer) and \
                   self._GetState(iiwa_num=2, context=context) == State.SKEWER:
                # if self._Timeout(0.8, iiwa_num=1, context=context):
                    # Go to skewering position
                    self._X1_desired(context, self.X1_skewered)
                    self._ChangeState(1, State.SKEWER, context)

            case State.SKEWER:
                # Move to skewer objects
                if self._AtPose(context, iiwa_num=1, X=self.X1_skewered, tol=0.15):
                    self._ResetTimeout(context, iiwa_num=1)

                    if self._Timeout(2, iiwa_num=1, context=context):
                        # Go back to preskewer position
                        self._X1_desired(context, self.X1_post_skewered)
                        self._ChangeState(1, State.START, context)

            case _:
                raise RuntimeError("Unexpected state")

        output.set_value(self._X1_desired(context))

    def CalcDesiredPose2(self, context, output):
        state = self._GetState(iiwa_num=2, context=context)
        gripper_state = context.get_abstract_state(int(self._gripper_state_index)).get_value()
        
        match state:
            case State.START:
                # Go to initial position
                # if False:
                if self._AtPose(context, iiwa_num=2, X=self.X2_preskewer):
                    # Take in point clouds and process
                    pcds = []
                    for i in range(self.num_cameras):
                        pcds.append(self.GetInputPort(f"camera{i}_point_cloud").Eval(context))
                    pcd = ProcessPointCloud(
                        pcds, 
                        self._world_plant, 
                        self._world_context,
                        crop_lower=(-2, -2, 0),
                        crop_upper=(-0.2, 0.1, 0.2),
                        remove_plane=True
                    )

                    # Grasp selection of objects
                    X_desired = AntipodalCandidateGrasp(pcd)

                    if X_desired is None:
                        raise RuntimeError("No grasp found")
                    
                    print(f"Grasp pose found")
                    
                    # Update goal and state
                    self.X_Grasp = X_desired
                    self._X2_desired(context, self.X_PregraspGrasp @ self.X_Grasp)
                    self._ChangeState(2, State.PREGRASP, context)

            case State.PREGRASP:
                # Move to pose above object
                if self._AtPose(context, iiwa_num=2, X=self._X2_desired(context)):
                    self._ResetTimeout(context, iiwa_num=2)

                    if self._Timeout(2, iiwa_num=2, context=context):
                        self._X2_desired(context, self.X_Grasp)
                        self._ChangeState(2, State.GRASP, context)

                # Timeout after x seconds and retry
                elif self._Timeout(3, iiwa_num=2, context=context):
                    self._X2_desired(context, self.X2_preskewer)
                    self._ChangeState(2, State.START, context)

            case State.GRASP:
                # Move to objects
                if self._AtPose(context, iiwa_num=2, X=self._X2_desired(context)):
                    self._ResetTimeout(context, iiwa_num=2)

                    # Move to next state after the gripper has some time to close
                    if self._Timeout(0.5, iiwa_num=2, context=context):
                        # Close gripper if is open, and reset timeout
                        if gripper_state == self.gripper_open:
                            self._ChangeGripper(context, self.gripper_closed)
                        else:
                            self._X2_desired(context, self.X2_preskewer)
                            self._ChangeState(2, State.GET_TO_SKEWER_POSITION, context)

                # Timeout after x seconds and retry
                elif self._Timeout(3, iiwa_num=2, context=context):
                    self._X2_desired(context, self.X2_preskewer)
                    self._ChangeState(2, State.START, context)

            case State.GET_TO_SKEWER_POSITION:
                # Move to skewer position and maintain
                if self._AtPose(context, iiwa_num=2, X=self.X2_preskewer):
                    self._ResetTimeout(context, iiwa_num=2)

                    if self._Timeout(2, iiwa_num=2, context=context):
                        self._ChangeState(2, State.SKEWER, context)

            case State.SKEWER:
                # Wait until iiwa1 skewers
                if self._AtPose(context, iiwa_num=1, X=self.X1_skewered, tol=0.15):
                    if self._Timeout(0.5, iiwa_num=2, context=context):
                        # Open gripper if is closed, and reset timeout
                        if gripper_state == self.gripper_closed:
                            self._ChangeGripper(context, self.gripper_open)
                        else:
                            # Move to next state
                            self._X2_desired(context, self.X2_released)
                            self._ChangeState(2, State.RELEASE, context)

            case State.RELEASE:
                # Move to release the object
                if self._AtPose(context, iiwa_num=2, X=self.X2_released):
                    self._ResetTimeout(context, iiwa_num=2)

                    # Move to next state after some time to settle
                    if self._Timeout(2.5, iiwa_num=2, context=context):
                        self._X2_desired(context, self.X2_preskewer)
                        self._ChangeState(2, State.START, context)

            case _:
                raise RuntimeError("Unexpected state")

        output.set_value(self._X2_desired(context))

        self._debug_count += 1
        if self._debug_count > self._debug_rate:
            self._debug_count = 0

    def CalcControlMode(self, context, output):
        state = self._GetState(iiwa_num=1, context=context)
        
        match state:
            case State.SKEWER:
                output.set_value(InputPortIndex(2))  # Use stiffness control
            case _:
                output.set_value(InputPortIndex(1))  # Use diff IK
    
    def CloseGripper(self, context, output):
        gripper_state = context.get_abstract_state(int(self._gripper_state_index)).get_value()

        output.SetFromVector(np.array([gripper_state, -gripper_state]))

    def _AtPose(self, 
                context: Context,
                iiwa_num: int,
                X: RigidTransform,
                tol: float = 0.05
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

            dist = self._PoseDifference(X_now, X)
            self._debug(f"iiwa{iiwa_num} goal dist: {dist}")

            return dist <= tol

        elif iiwa_num == 2:
            # Update plant
            self._plant2.SetPositions(self._plant_context2, self._iiwa2, self._q2_in.Eval(context))

            # Current pose of the end effector
            X_now = self._plant2.CalcRelativeTransform(
                self._plant_context2,
                self._plant2.world_frame(),
                self._G2
            )

            dist = self._PoseDifference(X_now, X)
            self._debug(f"iiwa{iiwa_num} goal dist: {dist}")

            return dist <= tol

        else:
            raise RuntimeError("iiwa_num was invalid")
        
    def _ResetTimeout(self, context: Context, iiwa_num: int):
        if iiwa_num == 1 and not self.timer1_reset:
            self.iiwa1_timer = context.get_time()
            self.timer1_reset = True
        elif iiwa_num == 2 and not self.timer2_reset:
            self.iiwa1_timer = context.get_time()
            self.timer2_reset = True
        
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
            print(f"iiwa{iiwa_num} Timeout")
            self.iiwa1_timer = cur_time
            return True
        
        elif iiwa_num == 2 and cur_time - self.iiwa2_timer > duration:
            print(f"iiwa{iiwa_num} Timeout")
            self.iiwa2_timer = cur_time
            return True
        
        else:
            return False
        
    def _GetState(self,
                  iiwa_num: int,
                  context: Context
                  ) -> State:
        """
        Gets the state of the desired iiwa
        """
        if   iiwa_num == 1: return context.get_abstract_state(int(self._state1_index)).get_value()
        elif iiwa_num == 2: return context.get_abstract_state(int(self._state2_index)).get_value()
        else:               raise RuntimeError("iiwa_num was invalid")
        
    def _ChangeState(self,
                     iiwa_num: int,
                     state: State,
                     context: Context
                     ) -> None:
        """
        Changes the state of the desired iiwa and updates its time
        """
        cur_state = self._GetState(iiwa_num=iiwa_num, context=context)
        cur_time = context.get_time()
        if iiwa_num == 1:
            context.get_mutable_abstract_state(int(self._state1_index)).set_value(state)
            self.iiwa1_timer = cur_time
            self.timer1_reset = False

        elif iiwa_num == 2:
            context.get_mutable_abstract_state(int(self._state2_index)).set_value(state)
            self.iiwa2_timer = cur_time
            self.timer2_reset = False

        else:
            raise RuntimeError("iiwa_num was invalid")
        
        print(f"iiwa{iiwa_num} {cur_state} -> {state}")

    def _ChangeGripper(self, context, value):
        print(f"Gripper set to {value}")
        context.get_mutable_abstract_state(int(self._gripper_state_index)).set_value(value)
        
    def _PoseDifference(self, X1: RigidTransform, X2: RigidTransform) -> np.ndarray:
        """
        Computes the 6x1 vector representing the pose error from X1 to X2.
        """

        def vee(S: np.ndarray) -> np.ndarray:
            """vee operator for a 3x3 skew-symmetric matrix"""
            return np.array([-S[1, 2], S[0, 2], -S[0, 1]]).reshape(-1, 1)

        # Extract rotations from RPY representations
        R_1W = X1.rotation().matrix()
        R_2W = X2.rotation().matrix()

        # Rotation error between x1 and x2 (small angle approximation)
        R_21 = R_2W @ R_1W.T
        rot_error = np.linalg.norm(vee(1 / 2 * (R_21 - R_21.T)))

        # Position error between x1 and x2
        pos_error = np.linalg.norm(X1.translation() - X2.translation())

        return rot_error + pos_error
    
    def _X1_desired(self, context, value=None):
        if value is None: return context.get_abstract_state(int(self._X1_desired_index)).get_value()
        else:             
            context.get_mutable_abstract_state(int(self._X1_desired_index)).set_value(value)
            self._drawGrasp(value, f"{self.grasp_count}")
            self.grasp_count += 1
    
    def _X2_desired(self, context, value=None):
        if value is None: return context.get_abstract_state(int(self._X2_desired_index)).get_value()
        else:             
            context.get_mutable_abstract_state(int(self._X2_desired_index)).set_value(value)
            self._drawGrasp(value, f"{self.grasp_count}")
            self.grasp_count += 1

    def _drawGrasp(self, X_G: RigidTransform, prefix: str):    
        if self.meshcat is not None:
            AddMeshcatTriad(self.meshcat, X_PT=X_G, path=f"grasp{prefix}", length=0.15, opacity=0.2)

    def _debug(self, print_string):
        if self._debug_count == self._debug_rate:
            print(print_string)