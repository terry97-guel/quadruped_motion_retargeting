# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
from mujoco_mpc import agent as agent_lib

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pathlib

# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib
# from mjtg import MJPC_TASK_PATH
# from mjtg.io import MotionIO
from motion_menagerie.mann import MANN_BASE_PATH
from mujoco_viewer import MujocoViewer
from mjtools import get_site_id, plot_sphere, reset, reset_with_mocap, plot_robot

# from dynamic_mr_cfg import Go1Cfg as cfg
from dynamic_mr_cfg import cfg

# %%
from pathlib import Path
MJPC_TASK_PATH = Path(__file__).parent.parent/ 'motion_mjpc/build_py/mjpc/tasks'
MJPC_TASK_PATH.exists()

# %%
# model
robot_name = cfg.ROBOT.split("_")[0]
robot_name = robot_name.capitalize()
model_path = (
    MJPC_TASK_PATH/f"model_files/robots/Quadruped/{robot_name}/{cfg.ROBOT}.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# data
data = mujoco.MjData(model)
planner_run_per_step = 2

# %%
# agent
agent = agent_lib.Agent(task_id=f"{robot_name}Motion", model=model)
agent.set_mode(cfg.MOTION)

# %%
try:
    viewer.close()
except Exception:
    pass
viewer = MujocoViewer(
        model,data,mode='window',title="MPC",
        width=600,height=400,hide_menus=True
)
viewer._contacts = True
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = viewer._contacts
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = viewer._contacts

reset(model, data)
viewer.cam.lookat = data.qpos[:3]
viewer.render()

agent.set_mode(cfg.MOTION)

# %%
import json
with open(cfg.motion_info_from_path, "r") as f:
    motion_info = json.load(f)

motion_names = list(motion_info.keys())
motion_names.sort()

mode = motion_names.index(cfg.MOTION)

# %%
def get_mocap_id(mode):
    mocap_id = 0
    for mode_i in range(mode):
        motion_name = motion_names[mode_i]
        mocap_id += motion_info[motion_name]["length"]
    return mocap_id

mocap_id = get_mocap_id(mode)
mocap_id

# %%
foot_names_ls = ["FL_foot", "FR_foot", "RR_foot", "RL_foot"]

def get_contact(model, data, foot_names_ls):
    contact_ls = []
    for i, foot_name in enumerate(foot_names_ls):
        IS_CONTACT = False
        for contact in data.contact:
            if model.geom(contact.geom1).name == foot_name and model.geom(contact.geom2).name == "floor":
                IS_CONTACT = True
                break
            if model.geom(contact.geom2).name == foot_name and model.geom(contact.geom1).name == "floor":
                IS_CONTACT = True
                break
        contact_ls.append(IS_CONTACT)
    return np.array(contact_ls, dtype=np.bool)
        
# %%
dt=1/motion_info[cfg.MOTION]['fps']
reset_with_mocap(model, data, mocap_id, INIT_QVEL=True)
viewer.render()

# %%
agent.reset()
reset_with_mocap(model, data, mocap_id, INIT_QVEL=True)
viewer.render()

# rollout horizon
fps = motion_info[cfg.MOTION]['fps']
frame_num = motion_info[cfg.MOTION]['length']
motion_time = frame_num / fps
T = int(motion_time /model.opt.timestep)

extra_step_interval = int(3.0 / model.opt.timestep)
pass_step_interval = int(0.0 / model.opt.timestep)

# extra_step_interval = 0
# pass_step_interval = 1000

extra_step_multiplier = 2

# %%
# plot target only
target_time_array = np.arange(0, motion_time, model.opt.timestep)
for idx in range(len(target_time_array)):
    if idx % 10 == 0:    
        time_ = target_time_array[idx]
        agent.set_state(
            time=time_
        )
        data.mocap_pos = np.array(agent.get_state().mocap_pos).reshape(-1, 3)
        mujoco.mj_forward(model, data)
        viewer.render()

# %%
# trajectories
qpos_array = np.zeros((T, model.nq))
qvel_array = np.zeros((T, model.nv))
ctrl_array = np.zeros((T - 1, model.nu, ))
mpos_array = np.zeros((T, model.nmocap, 3))
time_array = np.zeros(T)
contact_array = np.zeros((T, 4))

# costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# rollout
# mujoco.mj_resetData(model, data)

# cache initial state
qpos_array[0] = data.qpos
qvel_array[0] = data.qvel
time_array[0] = data.time
contact_array[0] = get_contact(model, data, foot_names_ls)

# frames
frames = []
FPS = 1.0 / model.opt.timestep

t = 0
# %%
# simulate
from tqdm import tqdm

t_start = t

for t in tqdm(range(t_start, T - 1)):
  # set planner state
    agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata,
    )

    # run planner for num_steps
    if t<extra_step_interval and t>pass_step_interval:
        planner_run_per_step_ = planner_run_per_step * extra_step_multiplier
    else:
        planner_run_per_step_ = planner_run_per_step

    if planner_run_per_step_ > 1:
        num_steps = int(planner_run_per_step_)
        for _ in range(num_steps):
            agent.planner_step()
    else:
        steps_every = int(1 / planner_run_per_step_)
        if t % steps_every == 0:
            agent.planner_step()

    data.ctrl = agent.get_action()
    ctrl_array[t] = data.ctrl

    # get costs
    cost_total[t] = agent.get_total_cost()
    # for i, c in enumerate(agent.get_cost_term_values().items()):
    #     cost_terms[i, t] = c[1]

    # plot target
    data.mocap_pos = np.array(agent.get_state().mocap_pos).reshape(-1, 3)
    # step
    mujoco.mj_step(model, data)

    # cache
    qpos_array[t + 1] = data.qpos
    qvel_array[t + 1] = data.qvel
    mpos_array[t + 1] = data.mocap_pos
    time_array[t + 1] = data.time
    contact_array[t + 1] = get_contact(model, data, foot_names_ls)

    # render
    if t%10==0:
        # viewer.cam.lookat = data.qpos[:3]
        # viewer.cam.lookat = data.mocap_pos[0,:3]
        viewer.cam.distance = 2.0
        viewer.cam.lookat = data.qpos[:3]
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -2
        viewer.render()
        for foot_i in range(4):
            foot_rgba = [1,0,0,0.5] if contact_array[t+1][foot_i] else [0,1,0,0.5]
            foot_name = foot_names_ls[foot_i]
            plot_sphere(viewer, p=data.site_xpos[get_site_id(model, foot_name+"_site")], r=0.05, rgba=[1,0,0,0.5])
            
        viewer.render()

# %%
viewer._paused = True
viewer.cam.distance = 2.0
viewer.cam.azimuth = 90

data.mocap_pos[:] = 10
plot_robot(viewer=viewer, model=model, data=data, qpos_array=qpos_array, lookat_site_idr=get_site_id(model, "trunk_site"), sphere_site_ids=[get_site_id(model, foot_name+"_site") for foot_name in foot_names_ls], PLOT_EVERY=10)

 # %%
for _ in range(1):
    height_list = []
    time_ = 0
    for idx, qpos in enumerate(qpos_array):
        if idx % 2 == 0:
            agent.set_state(
                time=time_
            )
            data.mocap_pos = np.array(agent.get_state().mocap_pos).reshape(-1, 3)
            data.qpos = qpos
            mujoco.mj_forward(model, data)
            viewer.render()
        
        height_list.append(data.qpos[2])
        time_ += model.opt.timestep
    print(max(height_list))

# %%
# plot ctrl_array
plt.figure()
plt.plot(ctrl_array)
plt.legend([f"ctrl_{i}" for i in range(model.nu)])

# %%
# plot target only
target_time_array = np.arange(0, motion_time, model.opt.timestep)
for idx in range(len(target_time_array)):
    if idx % 10 == 0:    
        time_ = target_time_array[idx]
        agent.set_state(
            time=time_
        )
        data.mocap_pos = np.array(agent.get_state().mocap_pos).reshape(-1, 3)
        mujoco.mj_forward(model, data)
        viewer.render()

# %%
from mjtools import qpos_index_from_names
from motion_menagerie import get_MR_json_path, MotionIO

motion_io = MotionIO(model, data, viewer).set_qpos(qpos_array)
motion_io.smart_export_xml(cfg.MOTION_BASE_PATH, cfg.ROBOT, cfg.MOTION, cfg.MR, dt=model.opt.timestep, USE_FD=True)

# %%
qpos_array_save = qpos_array
qpos_array_indexed = qpos_array_save[:, qpos_index_from_names(model, cfg.crl_joint_orders)]
motion_json_path = get_MR_json_path(cfg.MOTION_BASE_PATH, cfg.ROBOT, cfg.MOTION, cfg.MR)
motion_json_path = motion_io.export_json(motion_json_path, cfg.foot_info_dict, qpos_array_indexed, dt=model.opt.timestep)


# %%
def output_amp_motion(frames, out_filename, motion_weight, frame_duration):
  from pathlib import Path
  Path(out_filename).parent.mkdir(parents=True, exist_ok=True)
  with open(out_filename, "w") as f:
    f.write("{\n")
    f.write("\"LoopMode\": \"Wrap\",\n")
    f.write("\"FrameDuration\": " + str(frame_duration) + ",\n")
    f.write("\"EnableCycleOffsetPosition\": true,\n")
    f.write("\"EnableCycleOffsetRotation\": true,\n")
    f.write("\"MotionWeight\": " + str(motion_weight) + ",\n")
    f.write("\n")

    f.write("\"Frames\":\n")
    f.write("[")
    for i in range(frames.shape[0]):
      curr_frame = frames[i]

      if i != 0:
        f.write(",")
      f.write("\n  [")
      for j in range(frames.shape[1]):
        curr_val = curr_frame[j]
        if j != 0:
          f.write(", ")
        f.write("%.5f" % curr_val)

      f.write("]")

    f.write("\n]")
    f.write("\n}")

  return True

qpos_amp = qpos_array.copy()
qpos_amp[:, [3,4,5,6]] = qpos_array[:, [4,5,6,3]]

output_amp_motion(qpos_amp, f"{cfg.ROBOT}_{cfg.MOTION}.txt", motion_weight=1, frame_duration= model.opt.timestep)

# %%
