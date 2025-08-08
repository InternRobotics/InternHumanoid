import sys
import time
import collections
import yaml
import torch
import numpy as np
import argparse
import mujoco
import mujoco.viewer
import pickle
import onnxruntime
from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Dict, Tuple, List
from dataclasses import dataclass
from pynput import keyboard

SIMULATION_DURATION = 600.0  # sec
SIMULATION_DT = 0.002  # sec
CONTROL_DECIMATION = 10  # steps


def load_onnx_policy(path, device="cuda:0"):
    model = onnxruntime.InferenceSession(path)

    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device=device)

    return run_inference


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


# ? >>> Control
@dataclass
class ControlState:
    # current values
    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_vel: float = 0.0
    height: float = 0.74
    exit_flag: bool = False

    # limits
    range_map: Dict[str, Tuple[float, float]] = None
    # step sizes
    step_map: Dict[str, float] = None
    # key bindings: key -> (field, +delta) ;
    keymap: Dict[str, Tuple[str, float]] = None

    def update(self, pressed: Dict[str, bool]) -> None:
        for key, is_down in pressed.items():
            if not is_down:
                continue
            if key in self.keymap:
                field, delta = self.keymap[key]
                step = self.step_map.get(field, 0.0) * (1 if delta > 0 else -1)
                new_val = getattr(self, field) + step
                lo, hi = self.range_map[field]
                setattr(self, field, clamp(new_val, lo, hi))
            elif key == "q":
                self.exit_flag = True

        def zero_if_both_up(neg_key: str, pos_key: str, field: str):
            if not pressed.get(neg_key, False) and not pressed.get(pos_key, False):
                setattr(self, field, 0.0)

        zero_if_both_up("s", "w", "x_vel")
        zero_if_both_up("d", "a", "y_vel")
        zero_if_both_up("r", "e", "yaw_vel")


class KeyboardHandler:
    def __init__(self, keys: List[str]):
        self.pressed = {k: False for k in keys}

    def on_press(self, key):
        try:
            c = key.char
            if c in self.pressed:
                self.pressed[c] = True
        except Exception:
            pass  # ignore special keys

    def on_release(self, key):
        try:
            c = key.char
            if c in self.pressed:
                self.pressed[c] = False
        except Exception:
            pass


# ? <<< Control


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ]
    )


def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


def discretize_speed(speeds: np.ndarray, K: float) -> np.ndarray:
    half = K * 0.5
    return np.floor((speeds + half) / K) * K


def compute_observation(d, default_angles, action, cmd, height_cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7 : 7 + n_joints].copy()
    dqj = d.qvel[6 : 6 + n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    # Handle default angles padding
    if len(default_angles) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[: len(default_angles)] = default_angles
    else:
        padded_defaults = default_angles[:n_joints]

    ang_vel_scale = 0.5
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    cmd_scale = [2.0, 2.0, 0.5]

    # Scale the values
    qj_scaled = (qj - padded_defaults) * dof_pos_scale
    dqj_scaled = dqj * dof_vel_scale
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * ang_vel_scale

    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + len(action)

    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * np.array(cmd_scale, np.float32)
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10 : 10 + n_joints] = qj_scaled
    single_obs[10 + n_joints : 10 + 2 * n_joints] = dqj_scaled
    single_obs[10 + 2 * n_joints : 10 + 2 * n_joints + len(action)] = action

    return single_obs, single_obs_dim


def main():
    # Load configuration
    xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/urdf/g1_29dof_heavy_payload.xml"
    ckpt_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/flomo.onnx"
    info_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/flomo.pkl"

    # --- Keyboard / Control ---
    control = ControlState(
        range_map={
            "x_vel": (-0.8, 1.0),
            "y_vel": (-0.5, 0.5),
            "yaw_vel": (-0.8, 0.8),
            "height": (0.34, 0.74),
        },
        step_map={"x_vel": 0.1, "y_vel": 0.1, "yaw_vel": 0.1, "height": 0.01},
        keymap={
            # move
            "w": ("x_vel", +1),
            "s": ("x_vel", -1),
            "a": ("y_vel", +1),
            "d": ("y_vel", -1),
            "e": ("yaw_vel", +1),
            "r": ("yaw_vel", -1),
            # height
            "x": ("height", +1),
            "z": ("height", -1),
            # "q" handled in update() to exit
        },
    )
    kb = KeyboardHandler(keys=list(control.keymap.keys()) + ["q"])
    listener = keyboard.Listener(on_press=kb.on_press, on_release=kb.on_release)
    listener.daemon = True
    listener.start()

    # Load robot model
    info = pickle.load(open(info_path, "rb"))
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT
    policy = load_onnx_policy(ckpt_path, device="cpu")

    NUM_ACTIONS = info["NUM ACTIONS"]
    NUM_OBS = info["NUM OBS"]
    LEN_HISTORY = info["LEN HISTORY"]
    STIFFNESS = info["STIFFNESS"]
    DAMPING = info["DAMPING"]
    DEFAULT_JOINT_ANGLES = info["DEFAULT JOINT ANGLES"]
    ACTION_SCALE = info["ACTION SCALE"]
    ACTION_DOF_NAMES = [
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_roll",
        "waist_pitch",
    ]

    joint_names = [m.joint(i).name for i in range(m.njnt)]
    print(joint_names)

    # Check number of joints
    n_joints = d.qpos.shape[0] - 7

    # Initialize variables
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    default_dof_pos = np.array(
        [DEFAULT_JOINT_ANGLES[d_name + "_joint"] for d_name in ACTION_DOF_NAMES], dtype=np.float32
    )
    target_dof_pos = np.array(
        [DEFAULT_JOINT_ANGLES[d_name + "_joint"] for d_name in ACTION_DOF_NAMES], dtype=np.float32
    )
    kps = np.array(
        [next(STIFFNESS[k] for k in STIFFNESS if k in d_name) for d_name in ACTION_DOF_NAMES], dtype=np.float32
    )
    kds = np.array([next(DAMPING[k] for k in DAMPING if k in d_name) for d_name in ACTION_DOF_NAMES], dtype=np.float32)

    cmd = np.zeros(3, dtype=np.float32)
    height_cmd = 0.74
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d, default_dof_pos, action, cmd, height_cmd, n_joints)

    obs_history = collections.deque(maxlen=LEN_HISTORY)
    for _ in range(LEN_HISTORY):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))

    # Prepare full observation vector
    obs = np.zeros(NUM_OBS, dtype=np.float32)

    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < SIMULATION_DURATION and not control.exit_flag:

            control.update(kb.pressed)

            step_start = time.time()

            lower_idx = np.concatenate([np.arange(0, 12), np.arange(13, 15)])
            upper_idx = np.concatenate([np.arange(12, 13), np.arange(15, 29)])

            # Control leg joints with policy
            leg_tau = pd_control(
                target_dof_pos[:14],
                d.qpos[7 + lower_idx],
                kps[:14],
                np.zeros_like(kps[:14]),
                d.qvel[6 + lower_idx],
                kds[:14],
            )

            d.ctrl[lower_idx] = leg_tau

            # Keep other joints at zero positions if they exist
            if n_joints > NUM_ACTIONS:
                arm_kp = 40.0
                arm_kd = 1.0
                arm_target_positions = np.zeros(n_joints - NUM_ACTIONS, dtype=np.float32)

                arm_tau = pd_control(
                    arm_target_positions,
                    d.qpos[7 + upper_idx],
                    np.ones(n_joints - NUM_ACTIONS) * arm_kp,
                    np.zeros(n_joints - NUM_ACTIONS),
                    d.qvel[6 + upper_idx],
                    np.ones(n_joints - NUM_ACTIONS) * arm_kd,
                )

                if d.ctrl.shape[0] > NUM_ACTIONS:
                    d.ctrl[upper_idx] = arm_tau

            # Step physics
            mujoco.mj_step(m, d)

            counter += 1
            if counter % CONTROL_DECIMATION == 0:
                # Update observation
                cmd[0] = control.x_vel
                cmd[1] = control.y_vel
                cmd[2] = control.yaw_vel
                height_cmd = control.height

                cmd[:3] = discretize_speed(cmd[:3], 0.1)

                cmd *= (np.linalg.norm(cmd[:3]) > 0.1).repeat(3)

                single_obs, _ = compute_observation(d, default_dof_pos, action, cmd, height_cmd, n_joints)

                obs_history.append(single_obs)

                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()

                # Transform action to target_dof_pos
                target_dof_pos = action * ACTION_SCALE + default_dof_pos[:14]

            # Sync viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
