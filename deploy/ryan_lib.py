import booster_robotics_sdk_python as B
from functools import partial
from time import sleep
from easydict import EasyDict
import rp
import math


def set_global(name, value):
    globals()[name] = value


@rp.globalize_locals
def _init_communication() -> None:
    if not "INITIALIZED" in globals():
        B.ChannelFactory.Instance().Init(0)

        # low_cmd_publisher = B1LowCmdPublisher()
        # low_cmd_publisher.InitChannel()
        # low_cmd = LowCmd()

        client = B.B1LocoClient()
        client.Init()

        handler = partial(set_global, "low_state")  # Low Level State
        low_state_subscriber = B.B1LowStateSubscriber(handler)
        low_state_subscriber.InitChannel()

        # Low-level joint control setup
        low_cmd_publisher = B.B1LowCmdPublisher()
        low_cmd_publisher.InitChannel()
        motor_cmds = [B.MotorCmd() for _ in range(B.B1JointCnt)]
        set_global("low_cmd_publisher", low_cmd_publisher)
        set_global("motor_cmds", motor_cmds)

        INITIALIZED = True
    else:
        rp.fansi_print("_init_communication: Already initialized", 'yellow italic')

def tick(): sleep(.01)

# Mode control
def set_mode_custom (): return client.ChangeMode(B.RobotMode.kCustom ) # Goes limp, allows low level control
def set_mode_prepare(): return client.ChangeMode(B.RobotMode.kPrepare) # Stand ready position
def set_mode_walking(): return client.ChangeMode(B.RobotMode.kWalking) # For movement and dancing. WARNING: Never do this when the robot is on the stand!
def set_mode_damping(): return client.ChangeMode(B.RobotMode.kDamping) # Damped/compliant mode

# Dance moves - must be in walking mode
def do_new_years_dance(): return client.Dance(B.DanceId.kNewYear      ) # New Year dance
def do_nezha_dance    (): return client.Dance(B.DanceId.kNezha        ) # Nezha dance
def do_future_dance   (): return client.Dance(B.DanceId.kTowardsFuture) # Towards Future dance
def stop_dance        (): return client.Dance(B.DanceId.kStop         ) # Stop dancing

# Movement control - must be in walking mode
def move(x=0.0, y=0.0, z=0.0): return client.Move(x, y, z) # x=forward/back, y=left/right, z=rotate
def stop_movement(): return move(0, 0, 0)
def walk_forward (speed=0.8): return move(speed, 0, 0 )
def walk_backward(speed=0.2): return move(-speed, 0, 0)
def walk_left    (speed=0.2): return move(0, speed, 0 )
def walk_right   (speed=0.2): return move(0, -speed, 0)
def rotate_left  (speed=0.2): return move(0, 0, speed )
def rotate_right (speed=0.2): return move(0, 0, -speed)

# Head control - must be in walking mode
def rotate_head(pitch=0.0, yaw=0.0): return client.RotateHead(pitch, yaw)
def head_look_down (): return rotate_head(1.0, 0.0   )
def head_look_up   (): return rotate_head(-0.3, 0.0  )
def head_look_left (): return rotate_head(0.0, 0.785 )  # 45 degrees
def head_look_right(): return rotate_head(0.0, -0.785)  # 45 degrees
def head_center    (): return rotate_head(0.0, 0.0   )

# Arm/hand control - must be in walking mode
def handwave      (): return client.WaveHand (B.kHandOpen ) # Wave hand open
def stop_handwave (): return client.WaveHand (B.kHandClose) # Wave hand close
def handshake     (): return client.Handshake(B.kHandOpen ) # Start handshake motion
def stop_handshake(): return client.Handshake(B.kHandClose) # End handshake motion

# Get up / lie down - must be in prepare mode. WARNING: It's very clumsy!
def lie_down(): return client.LieDown()  # Makes robot lie down
def get_up  (): return client.GetUp  ()  # Makes robot stand up from lying position

# Pose control - bare bones
def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def get_current_posture():
    """Get current hand postures. Works in custom mode."""
    left_transform = B.Transform()
    right_transform = B.Transform()

    res_left = client.GetFrameTransform(B.Frame.kBody, B.Frame.kLeftHand, left_transform)
    res_right = client.GetFrameTransform(B.Frame.kBody, B.Frame.kRightHand, right_transform)

    if res_left != 0 or res_right != 0:
        raise RuntimeError(f"GetFrameTransform failed. Left: {res_left}, Right: {res_right}")

    # Convert to postures (position AND orientation)
    left_posture = B.Posture()
    left_posture.position = left_transform.position
    left_quat = left_transform.orientation
    left_roll, left_pitch, left_yaw = quaternion_to_euler(left_quat.x, left_quat.y, left_quat.z, left_quat.w)
    left_posture.orientation = B.Orientation(left_roll, left_pitch, left_yaw)

    right_posture = B.Posture()
    right_posture.position = right_transform.position
    right_quat = right_transform.orientation
    right_roll, right_pitch, right_yaw = quaternion_to_euler(right_quat.x, right_quat.y, right_quat.z, right_quat.w)
    right_posture.orientation = B.Orientation(right_roll, right_pitch, right_yaw)

    return {
        B.B1HandIndex.kLeftHand: left_posture,
        B.B1HandIndex.kRightHand: right_posture
    }

def set_posture(posture_dict, duration_ms=2000):
    """Set hand postures. Works in walking mode."""
    res_left = client.MoveHandEndEffectorV2(posture_dict[B.B1HandIndex.kLeftHand], duration_ms, B.B1HandIndex.kLeftHand)
    res_right = client.MoveHandEndEffectorV2(posture_dict[B.B1HandIndex.kRightHand], duration_ms, B.B1HandIndex.kRightHand)

    if res_left != 0 or res_right != 0:
        raise RuntimeError(f"MoveHandEndEffectorV2 failed. Left: {res_left}, Right: {res_right}")

    return res_left, res_right

def print_posture(posture_dict):
    """Print posture data in human-readable format."""
    for hand_index, posture in posture_dict.items():
        hand_name = "Left" if hand_index == B.B1HandIndex.kLeftHand else "Right"
        pos = posture.position
        orient = posture.orientation
        print(f"{hand_name} Hand:")
        print(f"  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
        print(f"  Orientation: roll={orient.roll:.3f}, pitch={orient.pitch:.3f}, yaw={orient.yaw:.3f}")

_init_communication()
