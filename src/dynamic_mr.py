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

# %%
from mujoco_mpc import agent as agent_lib


# %%
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
from mjtools import get_site_id, plot_sphere, reset


# %%
from pathlib import Path
MJPC_TASK_PATH = Path(__file__).parent/ 'quadruped_motion_mjpc/build_py/mjpc/tasks'
MJPC_TASK_PATH.exists()

# %%
# Choose ROBOT and MOTION here
ROBOT = "go2"
MOTION = "D1_009_KAN01_002"
# MOTION = "D1_025_KAN01_001"
# MOTION = "D1_049_KAN01_001"

name = str.replace(MOTION, "D1_","")
name = str.replace(name, "KAN01_","")

# %%
ROBOT = ROBOT.lower()
# model
model_path = (
    MJPC_TASK_PATH/f"{ROBOT}Motion/{ROBOT}_task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# data
data = mujoco.MjData(model)
planner_run_per_step = 1

# agent
agent = agent_lib.Agent(task_id=f"{ROBOT.capitalize()}Motion", model=model)

try:
    viewer.close()
except Exception:
    pass
viewer = MujocoViewer(
        model,data,mode='window',title="MPC",
        width=1200,height=800,hide_menus=True
)

# %%
# MOTION = "PushUp"
agent.set_mode(name)

# %%
MOTION_INFO_LIST = [
    {"name": "Stand", "length": 240, "fps": 30},
    {"name": "PushUp", "length": 240, "fps": 30},
    {"name": "Bound", "length": 240, "fps": 30},
    {"name": "BoundFast", "length": 240, "fps": 30},
    {"name": "BoundFaster", "length": 240, "fps": 30},
    {"name": "009_001", "length": 546, "fps": 60},
    {"name": "010_004", "length": 358, "fps": 60},
    {"name": "047z_005", "length": 2749, "fps": 60},
    {"name": "010_002", "length": 153, "fps": 60},
    {"name": "010_003", "length": 201, "fps": 60},
    {"name": "007_001", "length": 499, "fps": 60},
    {"name": "009_002", "length": 306, "fps": 60},
    {"name": "025_001", "length": 371, "fps": 60},
    {"name": "049_001", "length": 385, "fps": 60},
    # {"name": "a1_stand", "length":499, "fps": 50}
]

MOTION_INDEX_DICT = {}
for i, info in enumerate(MOTION_INFO_LIST):
    MOTION_INDEX_DICT[info["name"]] = i
mode = MOTION_INDEX_DICT[name]

# %%
def get_mocap_id(mode):
    mocap_id = 0
    for prev_mode in range(mode):
        mocap_id += MOTION_INFO_LIST[prev_mode]["length"]
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
dt=1/MOTION_INFO_LIST[mode]['fps']
reset(model, data, mocap_id, INIT_QVEL=True)
viewer.render()

# %%
agent.reset()
reset(model, data, mocap_id, INIT_QVEL=True)
viewer.render()

# rollout horizon
fps = MOTION_INFO_LIST[mode]['fps']
frame_num = MOTION_INFO_LIST[mode]['length']
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
from tqdm.notebook import tqdm

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
    for i, c in enumerate(agent.get_cost_term_values().items()):
        cost_terms[i, t] = c[1]

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
