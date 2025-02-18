# %%
import numpy as np
import mujoco
from mujoco_viewer import MujocoViewer

from robot_menagerie import ASSET_XML_DICT
from mjtools import reset
from smr.constants import QUADRUPED_PHASE_OFFSET as PHASE_OFFSET
from smr.task.Quadruped.info import QuadrupedSMRInfo

from matplotlib import pyplot as plt
import argparse

from smr.agent import SMR
from smr.ik_target_holder import IKTargetHolder, TimeStampedTarget

# from quadruped_walking_config import CustomCfg as cfg

# from kinematic_mr_cfg import Go1Cfg as cfg
from kinematic_mr_cfg import cfg

if cfg.ROBOT == "go1_task":
    qpos0 = np.array([0, 0.9, -1.8]*4)
else:
    qpos0 = np.zeros(12)
# %%
cfg.MOTION = "backflip8"
PI = np.pi

# Initalize Mujoco Model
xml_path = ASSET_XML_DICT[cfg.ROBOT]

model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
data  = mujoco.MjData(model)
reset(model,data)

# Initalize Viewer
try:
    viewer.close()
except Exception:
    pass
viewer = MujocoViewer(
model,data,mode='window',title="MPC",
width=400,height=300,hide_menus=True
)

viewer.cam.lookat = data.qpos[:3]
viewer.render()

# %%
qpos_list = []
contact_list = []

# %%
# viewer._paused = True
viewer.cam.azimuth = 90
viewer.cam.elevation = -10
viewer.render()
viewer._contacts = True
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = viewer._contacts
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = viewer._contacts


# %%
spread = -0.05

# %% 
x = 0.3
qpos = np.array([
    0., 0., 0.245,
    1., 0., 0., 0.,
    -spread, x*0.2, -x*0.4,
    +spread, x*0.2, -x*0.4,
    -spread, x*0.2, -x*0.4,
    +spread, x*0.2, -x*0.4])

qpos[7:] += qpos0
data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

for _ in range(2):
    qpos_list.append(qpos)
    contact_list.append([True, True, True, True])

# %% 
x = 1.0
qpos = np.array([
    0., 0., 0.20,
    1., 0., 0., 0.,
    -spread, x*0.2, -x*0.4,
    +spread, x*0.2, -x*0.4,
    -spread, x*0.2, -x*0.4,
    +spread, x*0.2, -x*0.4])

qpos[7:] += qpos0
data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

for _ in range(5):
    qpos_list.append(qpos)
    contact_list.append([True, True, True, True])


# %%
x = 0.5
theta = -np.pi/8
qpos = np.array([
    -0.03, 0., 0.27,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*0.5, x*0.7,
    +spread, -x*0.5, x*0.7,
    -spread, x*0.2, -x*0.4,
    +spread, x*0.2, -x*0.4])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

qpos_list.append(qpos)
contact_list.append([True, True, True, True])


# %%
x = 0.5
theta = -np.pi * 3/16
qpos = np.array([
    -0.14, 0., 0.37,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x/2, x,
    +spread, -x/2, x,
    -spread, 0, x *0.5,
    +spread, 0, x *0.5])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, True, True])

# %%
x = 0.5
theta = -np.pi  * 5/16
qpos = np.array([
    -0.20, 0., 0.44,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x/2, x,
    +spread, -x/2, x,
    -spread, 0.5*x, x,
    +spread, 0.5*x, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 7/16
qpos = np.array([
    -0.22, 0., 0.50,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x/2, x,
    +spread, -x/2, x,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 9/16
qpos = np.array([
    -0.23, 0., 0.55,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x/2, x,
    +spread, -x/2, x,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 11/16
qpos = np.array([
    -0.24, 0., 0.59,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x/2, x,
    +spread, -x/2, x,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 13/16
qpos = np.array([
    -0.25, 0., 0.62,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x, -x,
    +spread, -x, -x,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 15/16
qpos = np.array([
    -0.26, 0., 0.64,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x, -x,
    +spread, -x, -x,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 17/16
qpos = np.array([
    -0.27, 0., 0.65,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*3.0, 0,
    +spread, -x*3.0, 0,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 19/16
qpos = np.array([
    -0.28, 0., 0.64,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*3.0, 0,
    +spread, -x*3.0, 0,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 21/16
qpos = np.array([
    -0.29, 0., 0.62,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*3.0, 0,
    +spread, -x*3.0, 0,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 23/16
qpos = np.array([
    -0.30, 0., 0.59,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*3.0, 0,
    +spread, -x*3.0, 0,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 25/16
qpos = np.array([
    -0.31, 0., 0.55,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*2.0, -x*0.5,
    +spread, -x*2.0, -x*0.5,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = 0.5
theta = -np.pi  * 27/16
qpos = np.array([
    -0.32, 0., 0.50,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x*2.0, -x*0.5,
    +spread, -x*2.0, -x*0.5,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])


# %%
x = 0.5
theta = -np.pi  * 29/16
qpos = np.array([
    -0.33, 0., 0.44,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, -x, -x*0.5,
    +spread, -x, -x*0.5,
    -spread, 0, x,
    +spread, 0, x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])


# %%
x = 0.5
theta = -np.pi  * 31/16
qpos = np.array([
    -0.34, 0., 0.37,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, 0, -x*0.5,
    +spread, 0, -x*0.5,
    -spread, 0, 0,
    +spread, 0, 0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])


# %%
theta = -np.pi * 14/7
x = 0.0
qpos = np.array([
    -0.35, 0., 0.30,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    -spread, x, -x,
    +spread, x, -x,
    -spread, x, -x,
    +spread, x, -x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

for _ in range(5):
    qpos_list.append(qpos)
    contact_list.append([True, True, True, True])

# %%
qpos_array = np.array(qpos_list)
contact_array = np.array(contact_list)

# %%
for i in range(len(qpos_array)):
    data.qpos = qpos_array[i]
    mujoco.mj_forward(model, data)
    for _ in range(10):
        viewer.render()

# %%
# viewer._paused = True
viewer.render()

# %%
reset(model,data)
home_site_pos = data.site_xpos.copy()

smr_info            = QuadrupedSMRInfo(model, data, only_foot=True)
smr_agent = SMR(model, data, smr_info)

# %%
time_array = np.arange(0, len(qpos_array), 1)
site_target_dict = {}

for idx, foot_id in enumerate(smr_info.id.foot_ids):
    sitename = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, foot_id)
    site_pos = data.site_xpos[foot_id].copy()
    site_target_dict[sitename] = []
    
    for qpos in qpos_array:
        data.qpos = qpos
        mujoco.mj_forward(model, data)
        site_pos = data.site_xpos[foot_id].copy()
        site_target_dict[sitename].append(site_pos.tolist())

for idx, hip_id in enumerate(smr_info.id.hip_ids):
    sitename = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, hip_id)
    site_pos = data.site_xpos[hip_id].copy()
    site_target_dict[sitename] = []
    
    for qpos in qpos_array:
        data.qpos = qpos
        mujoco.mj_forward(model, data)
        site_pos = data.site_xpos[hip_id].copy()
        site_target_dict[sitename].append(site_pos.tolist())

# %%
print(f"target site keys: {site_target_dict.keys()}")
time_stamped_target = TimeStampedTarget(time_array, site_target_dict, contact_array)
ik_target_holder = IKTargetHolder(model, data, smr_info).from_time_stamped_target(time_stamped_target)

# %%
# smr_agent.smr_info.size.foot_size = 0.022
smr_agent.smr_info.size.contact_point = 0.00

# %%
smr_agent.set_ik_target(ik_target_holder)
smr_agent.spatial_retarget(qpos_array, viewer=viewer)

# %%
for qpos in smr_agent.spatial_retarget_result:
    data.qpos = qpos
    mujoco.mj_forward(model, data)
    for _ in range(10):
        viewer.render()

# %%
# Save motion

# %%
from motion_menagerie import MotionIO, get_MR_json_path
qpos_array_save = smr_agent.spatial_retarget_result

# Save to xml file (for MJPC)
motion_io = MotionIO(model, data, viewer).set_qpos(qpos_array_save)
motion_io.smart_export_xml(cfg.MOTION_BASE_PATH, cfg.ROBOT, cfg.MOTION, cfg.MR, dt=0.05, USE_FD=True)

# %%
# %% 
spread = 0.05
theta_1 = 0.5
theta_2 = -0.4
qpos = np.array([
    0., 0., 0.30,
    1., 0., 0., 0.,
    -spread, theta_1, theta_2,
    spread, theta_1, theta_2,
    -spread, theta_1, theta_2,
    spread, theta_1, theta_2])

x = 0.5
qpos[7:] += qpos0
data.qpos = qpos
data.qvel[0] = 0
data.qacc[0] = 0
mujoco.mj_forward(model, data)
viewer.render()

for frame_i in range(500):
    data.ctrl = (qpos[7:] - data.qpos[7:])*10
    mujoco.mj_step(model, data)
    if frame_i % 50 == 0:
        viewer.render()

# %%
height_list = []
data.ctrl[:] = 0 
data.ctrl[[0,3,6,9]] = 0.0
data.ctrl[[1,4,7,10]] = -1.0
data.ctrl[[2,5,8,11]] = 1.0


for frame_i in range(500):
    mujoco.mj_step(model, data)
    if frame_i % 10 == 0:
        viewer.render()
    height_list.append(data.qpos[2].copy())

print(max(height_list))
# %% 
spread = 0.05
theta_1 = -0.3
theta_2 = 0
qpos = np.array([
    0., 0., 0.30,
    1., 0., 0., 0.,
    -spread, theta_1, theta_2,
    spread, theta_1, theta_2,
    -spread, theta_1, theta_2,
    spread, theta_1, theta_2])

x = 0.5
qpos[7:] += qpos0
data.qpos = qpos
data.qvel[0] = 0
data.qacc[0] = 0
mujoco.mj_forward(model, data)
viewer.render()

for frame_i in range(500):
    data.ctrl = (qpos[7:] - data.qpos[7:])*10
    mujoco.mj_step(model, data)
    if frame_i % 50 == 0:
        viewer.render()
# %%
height_list = []
data.ctrl[:] = 0 
data.ctrl[[0,3,6,9]] = 0.0
data.ctrl[[1,4,7,10]] = 1.0
data.ctrl[[2,5,8,11]] = 1.0


for frame_i in range(500):
    mujoco.mj_step(model, data)
    if frame_i % 10 == 0:
        viewer.render()
    height_list.append(data.qpos[2].copy())

print(max(height_list))


