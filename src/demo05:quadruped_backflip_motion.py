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

from kinematic_mr_cfg import Go1Cfg as cfg
# from kinematic_mr_cfg import cfg

if cfg.ROBOT == "go1_task":
    qpos0 = np.array([0, 0.9, -1.8]*4)
else:
    qpos0 = np.zeros(12)
# %%
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
width=300,height=200,hide_menus=True
)

viewer.cam.lookat = data.qpos[:3]
viewer.render()

# %%
qpos_list = []
contact_list = []

# %%
# viewer._paused = True
viewer.render()

# %% 
qpos = np.array([
    0., 0., 0.26,
    1., 0., 0., 0.,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0])
qpos[7:] += qpos0
data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

for _ in range(10):
    qpos_list.append(qpos)
    contact_list.append([True, True, True, True])

# %%
x = 0.25
qpos = np.array([
    0., 0., 0.22,
    1., 0., 0., 0.,
    0, x, -x,
    0, x, -x,
    0, x, -x,
    0, x, -x])
qpos[7:] += qpos0
data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([True, True, True, True])

# # %%
# x = 0.3
# qpos = np.array([
#     0., 0., 0.21,
#     1., 0., 0., 0.,
#     0, x, -x,
#     0, x, -x,
#     0, x, -x,
#     0, x, -x])
# qpos[7:] += qpos0

# data.qpos = qpos
# mujoco.mj_forward(model, data)
# viewer.render()
# for _ in range(5):
#     qpos_list.append(qpos)
#     contact_list.append([True, True, True, True])

# %%
x = 0.3
qpos = np.array([
    0., 0., 0.21,
    1., 0., 0., 0.,
    0, x, -x,
    0, x, -x,
    0, x, -x,
    0, x, -x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
for _ in range(5):
    qpos_list.append(qpos)
    contact_list.append([True, True, True, True])
# %%
x = 0.3
theta = -np.pi/30
qpos = np.array([
    -0.03, 0., 0.24,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x/2, 0,
    0, x/2, 0,
    0, x, -x,
    0, x, -x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([True, True, True, True])

# %%
x = -0.5
theta = -np.pi/8
qpos = np.array([
    -0.27, 0., 0.34,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x, -x,
    0, x, -x,
    0, x, -x,
    0, x, -x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, True, True])

# %%
x = -0.5
theta = -np.pi * 2/7
qpos = np.array([
    -0.30, 0., 0.43,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*1.5, -0,
    0, x*1.5, -0,
    0, 0, -x*1.0,
    0, 0, -x*1.0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, True, True])

# %%
x = -0.5
theta = -np.pi * 3/7
qpos = np.array([
    -0.45, 0., 0.45,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*2.0, -x,
    0, x*2.0, -x,
    0, -x *2/5, -x*1.2,
    0, -x *2/5, -x*1.2])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, True, True])


# %%
x = -0.5
theta = -np.pi * 4/7
qpos = np.array([
    -0.55, 0., 0.50,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*2.0, -x,
    0, x*2.0, -x,
    0, -x*1.2, -x*2.5,
    0, -x*1.2, -x*2.5])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, True, True])


# %%
x = -0.5
theta = -np.pi * 5/7
qpos = np.array([
    -0.60, 0., 0.53,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*2.0, -x,
    0, x*2.0, -x,
    0, -x, -x*2.0,
    0, -x, -x*2.0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = -0.5
theta = -np.pi * 6/7
qpos = np.array([
    -0.64, 0., 0.55,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*2.0, -x,
    0, x*2.0, -x,
    0, -x, -x*2.0,
    0, -x, -x*2.0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = -0.5
theta = -np.pi * 7/7
qpos = np.array([
    -0.68, 0., 0.56,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*1.5, x * 0.5,
    0, x*1.5, x * 0.5,
    0, -x, -x*2.0,
    0, -x, -x*2.0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])


# %%
x = -0.5
theta = -np.pi * 8/7
qpos = np.array([
    -0.72, 0., 0.55,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*1.0, x,
    0, x*1.0, x,
    0, -x, -x*2.0,
    0, -x, -x*2.0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = -0.5
theta = -np.pi * 9/7
qpos = np.array([
    -0.76, 0., 0.53,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*1.0, x,
    0, x*1.0, x,
    0, -x, -x*1.0,
    0, -x, -x*1.0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = -0.5
theta = -np.pi * 10/7
qpos = np.array([
    -0.80, 0., 0.50,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*1.0, x,
    0, x*1.0, x,
    0, -x, -x*0.5,
    0, -x, -x*0.5])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = -0.5
theta = -np.pi * 11/7
qpos = np.array([
    -0.84, 0., 0.45,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*0.5, 0,
    0, x*0.5, 0,
    0, -x*0.5, -0,
    0, -x*0.5, -0])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([False, False, False, False])

# %%
x = -0.5
theta = -np.pi * 12/7
qpos = np.array([
    -0.88, 0., 0.36,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*0.5, 0,
    0, x*0.5, 0,
    0, 0, 0.5*x,
    0, 0, 0.5*x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([True, True, False, False])

# %%
x = -0.5
theta = -np.pi * 13/7
qpos = np.array([
    -0.92, 0., 0.36,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x*0.5, 0,
    0, x*0.5, 0,
    0, 0, 0.5*x,
    0, 0, 0.5*x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()
qpos_list.append(qpos)
contact_list.append([True, True, False, False])

# %%
theta = -np.pi * 14/7
x = 0.25
qpos = np.array([
    -0.96, 0., 0.22,
    np.cos(theta/2), 0., np.sin(theta/2), 0.,
    0, x, -x,
    0, x, -x,
    0, x, -x,
    0, x, -x])
qpos[7:] += qpos0

data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

for _ in range(5):
    qpos_list.append(qpos)
    contact_list.append([True, True, True, True])

# %%
qpos = np.array([
    -0.96, 0., 0.26,
    1., 0., 0., 0.,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0])
qpos[7:] += qpos0
data.qpos = qpos
mujoco.mj_forward(model, data)
viewer.render()

for _ in range(3):
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
motion_io.smart_export_xml(cfg.MOTION_BASE_PATH, cfg.ROBOT, cfg.MOTION, cfg.MR, dt=0.06, USE_FD=True)

# %%