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
from mjtools import get_site_id, plot_sphere, reset, reset_with_mocap


# %%
from pathlib import Path
MJPC_TASK_PATH = Path(__file__).parent.parent/ 'quadruped_motion_mjpc/build/mjpc/tasks'
MJPC_TASK_PATH.exists()

# %%
# Choose ROBOT and MOTION here
ROBOT = "go2_task"
ROBOT_CLASS = "Quadruped"
MOTION = "D1_007_KAN01_001"

robot_name = ROBOT.split("_")[0]
robot_name = robot_name.capitalize()

# %%
ROBOT = ROBOT.lower()
# model
model_path = (
    MJPC_TASK_PATH/f"model_files/robots/Quadruped/{robot_name}/{ROBOT}.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# data
data = mujoco.MjData(model)
planner_run_per_step = 1

# %%
model.nmocap

# %%
# agent
agent = agent_lib.Agent(task_id=f"{robot_name}Motion", model=model)

# %%
agent.get_all_modes()

# %%
agent.get_cost_term_values()

# %%
agent.reset()

# %%
agent.planner_step()

# %%
agent.set_mode(MOTION)

# %%
try:
    viewer.close()
except Exception:
    pass
viewer = MujocoViewer(
        model,data,mode='window',title="MPC",
        width=1200,height=800,hide_menus=True
)

reset(model, data)
viewer.cam.lookat = data.qpos[:3]
viewer.render()

agent.set_mode(MOTION)
# %%
from motion_menagerie.mann_lifted import MANN_LIFTED_BASE_PATH as MOTION_BASE_PATH
motion_info_from_path = (MOTION_BASE_PATH/f"kinematic_processed_data/{ROBOT}/motion_info.json").resolve()

# %%
import json
with open(motion_info_from_path, "r") as f:
    motion_info = json.load(f)

motion_names = list(motion_info.keys())
motion_names.sort()

mode = motion_names.index(MOTION)

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
dt=1/motion_info[MOTION]['fps']
reset_with_mocap(model, data, mocap_id, INIT_QVEL=True)
viewer.render()

# %%
agent.reset()
reset_with_mocap(model, data, mocap_id, INIT_QVEL=True)
viewer.render()

# rollout horizon
fps = motion_info[MOTION]['fps']
frame_num = motion_info[MOTION]['length']
motion_time = frame_num / fps
T = int(motion_time /model.opt.timestep)

extra_step_interval = int(0.5 / model.opt.timestep)
extra_step_multiplier = 10

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
    if t<extra_step_interval:
        planner_run_per_step_ = planner_run_per_step * extra_step_multiplier
    else:
        planner_run_per_step_ = planner_run_per_step

    if planner_run_per_step_ > 1:
        num_steps = int(planner_run_per_step_)
        for _ in range(num_steps):
            agent.planner_step()
    else:
        num_steps = int(1 / planner_run_per_step_)
        if t % num_steps == 0:
            agent.planner_step()

    # if t % 10 == 0:
    #     agent.planner_step()

    # set ctrl from agent policy
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
        viewer.cam.lookat = data.mocap_pos[0,:3]
        for foot_i in range(4):
            foot_rgba = [1,0,0,0.5] if contact_array[t+1][foot_i] else [0,1,0,0.5]
            foot_name = foot_names_ls[foot_i]
            plot_sphere(viewer, p=data.site_xpos[get_site_id(model, foot_name+"_site")], r=0.05, rgba=[1,0,0,0.5])
            
        viewer.render()

# %%
for t in range(T):
    if t %10==0:
        viewer.cam.lookat = data.qpos[:3]
        data.qpos = qpos_array[t]
        data.mocap_pos = mpos_array[t]
        # viewer.cam.lookat = data.qpos[:3]
        mujoco.mj_forward(model, data)

        for foot_i, foot_name in enumerate(foot_names_ls):
            foot_rgba = [1,0,0,0.5] if contact_array[t][foot_i] else [0,1,0,0.5]
            plot_sphere(viewer, p=data.site_xpos[get_site_id(model, foot_name+"_site")], r=0.05, rgba=foot_rgba)
        viewer.render()
# # %%
# dt_sim = model.opt.timestep
# save_every = 1

# motionIO = MotionIO(model, data, viewer=viewer).set_qpos(qpos_array, qvel_array, mpos_array)

# output_name= MANN_BASE_PATH/f"processed_data/xml/{ROBOT}/{MOTION}.xml"
# output_name.parent.mkdir(parents=True, exist_ok=True)
# motionIO.export_xml(XML_PATH=output_name,  RENDER_EVERY=10, dt=dt_sim*save_every)

# # %%
# from mann import MANN_BASE_PATH
# output_name = MANN_BASE_PATH/f"processed_data/json/{ROBOT}/{MOTION}.json"
# motionIO.export_json(output_name, qpos_array, qvel_array, dt_sim)
