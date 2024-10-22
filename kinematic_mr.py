# %%
from mjtools import get_body_id, get_site_id, plot_sphere, reset, get_mr_info, R2quat, get_mujoco_contact_boolean, plot_quadruped_contact_schedule, IOU, calculate_foot_slip, get_vel_contact_boolean, quat2R
from smr.task.Quadruped.info import QuadrupedSMRInfo
from motion_menagerie.mann import MANN_BASE_PATH

from robot_menagerie import ASSET_XML_DICT

from smr.agent import SMR
from smr.ik_target_holder import IKTargetHolder
# from mjmr.motion_holder import IKTargetHolder

import numpy as np
import time
import mujoco
import mujoco_viewer
from matplotlib import pyplot as plt
import json


ROBOT = "go2_task"
TASK = f"{ROBOT}Motion"
MOTION = "D1_009_KAN01_002"
# MOTION = "D1_047z_KAN01_005"
# MOTION = "D1_009_KAN01_001"
# MOTION = "D1_010_KAN01_004"
PLOT = True

xml_path = ASSET_XML_DICT[ROBOT]
model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
data  = mujoco.MjData(model)
reset(model, data)

homepos = data.qpos[:3].copy()
home_site_pos = data.site_xpos.copy()

try:
    viewer.close()
except Exception:
    pass
viewer = mujoco_viewer.MujocoViewer(
   model,data,mode='window',title="MPC",
   width=1200,height=800,hide_menus=True
   )

viewer.cam.lookat = data.qpos[:3]
viewer.render()
# %%
# Constants
KEY_MAP = {
    'RightShoulder' : "FR_hip_site",
    'RightHand'     : "FR_foot_site",
    'LeftShoulder'  : "FL_hip_site",
    'LeftHand'      : "FL_foot_site",
    'RightUpLeg'    : "RR_hip_site",
    'RightFoot'     : "RR_foot_site",
    'LeftUpLeg'     : "RL_hip_site",
    'LeftFoot'      : "RL_foot_site",
}

class ARGS:
    name = MOTION
    name = str.replace(name, "D1_","")
    name = str.replace(name, "KAN01_","")
    
    contact_json_path = MANN_BASE_PATH/"data"/f"{MOTION}_contact.json"
    traj_json_path = MANN_BASE_PATH/"data"/f"{MOTION}_traj.json"
    if "a1" in ROBOT.lower():
        scale = np.array([0.8,1,0.8])
    elif "go2" in ROBOT.lower():
        scale = np.array([0.8,1.0,0.90])
    dt = 1/60
    max_frame = -1

args = ARGS
max_frame = args.max_frame

with open(args.traj_json_path) as f:
    traj_json = json.load(f)    

if max_frame is None or max_frame == -1:
    max_frame = len(traj_json)
else:
    traj_json = traj_json[:max_frame]

# %%
from smr.ik_target_holder import IKTargetHolder, TimeStampedTarget

smr_info  = QuadrupedSMRInfo(model, data, only_foot=True)
smr_agent = SMR(model, data, smr_info)
# %%
# read trajectory json
def read_traj_json(traj_json, PLOT):
    base_quat_list = []
    base_pos_list = []
    time_list = []
    global_keypoint_dict = {}

    # plot target trajectory
    for traj_ in traj_json:
        time_ = traj_[0]
        time_list.append(time_)
        traj_dict = traj_[1]

        for key, value in traj_dict.items():
            if 'quat' in key:
                base_quat_list.append(value)
                continue
            elif "base_pos" in key:
                base_pos_list.append(value)
                if PLOT: plot_sphere(viewer, value, r=0.01, rgba=[1,0,0,1], label = key)
                continue
            else:
                if PLOT: plot_sphere(viewer, value, r=0.01, rgba=[1,0,0,1], label = key)
                foot_name = KEY_MAP[key]

                if foot_name not in global_keypoint_dict.keys():
                    global_keypoint_dict[foot_name] = []
                global_keypoint_dict[foot_name].append(value)

        if PLOT: 
            viewer.cam.lookat = base_pos_list[-1]
            viewer.render()

    base_quat_array = np.array(base_quat_list)
    base_pos_array = np.array(base_pos_list)
    assert base_quat_array.shape[0] == max_frame
    assert base_pos_array.shape[0] == max_frame

    for key,value in global_keypoint_dict.items():
        global_keypoint_dict[key] = np.array(value)
        assert len(global_keypoint_dict[key]) == max_frame

    time_array = np.array(time_list)
    time_array = time_array-time_array[0]
    
    return base_quat_array, base_pos_array, global_keypoint_dict, time_array

base_quat_array, base_pos_array, global_keypoint_dict, time_array = read_traj_json(traj_json, PLOT=PLOT)
# %%
# get local foot position
def get_foot_pos_dict(global_keypoint_dict, base_quat_array, PLOT=True):
    foot_pos_dict = {}
    foot_offset = np.array([-0.04, 0, 0])

    reset(model, data)
    hip_home_pos_dict = dict(
        FL_hip_site = data.site_xpos[smr_info.id.FL_thigh_site].copy(),
        FR_hip_site = data.site_xpos[smr_info.id.FR_thigh_site].copy(),
        RL_hip_site = data.site_xpos[smr_info.id.RL_thigh_site].copy(),
        RR_hip_site = data.site_xpos[smr_info.id.RR_thigh_site].copy(),
    )
    
    for frame_i in range(max_frame):
        for id_ in range(4):
            foot_name = smr_info.foot_names[id_]
            hip_name = smr_info.hip_names[id_]
            
            hip_pos = global_keypoint_dict[hip_name][frame_i]
            foot_pos = global_keypoint_dict[foot_name][frame_i]
            base_quat = base_quat_array[frame_i]
            base_R = quat2R(base_quat)

            if foot_name not in foot_pos_dict.keys():
                foot_pos_dict[foot_name] = []

            hip_to_foot_global = foot_pos - hip_pos
            hip_to_foot_local = np.dot(base_R.T, hip_to_foot_global)
            hip_to_foot_local += foot_offset 
            foot_pos_dict[foot_name].append(hip_to_foot_local * args.scale + smr_info.home_thigh_xpos[id_])
            # foot_pos_dict[foot_name].append(hip_to_foot_local * args.scale + hip_home_pos_dict[hip_name])

    for key,value in foot_pos_dict.items():
        foot_pos_dict[key] = np.array(value)
        assert len(foot_pos_dict[key]) == max_frame

    if PLOT:
        for frame_i in range(max_frame):
            for key,value in foot_pos_dict.items():
                plot_sphere(viewer, value[frame_i], r=0.01, rgba=[1,0,0,1], label = key)
            
            for key,value in hip_home_pos_dict.items():
                plot_sphere(viewer, value, r=0.01, rgba=[1,0,0,1], label = key)

            plot_sphere(viewer, homepos, r=0.01, rgba=[1,0,0,1], label = "base")
            viewer.cam.lookat = homepos[:3]
            viewer.render()
    return foot_pos_dict

foot_pos_dict = get_foot_pos_dict(global_keypoint_dict, base_quat_array, PLOT=PLOT)

# %%
# site_target_dict to build ik_target
def get_site_target_dict(foot_pos_dict):
    site_target_dict = foot_pos_dict

    for hip_id in smr_info.id.hip_ids:
        sitename = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, hip_id)
        site_target_dict[sitename] = np.tile(home_site_pos[hip_id], (max_frame,1)).tolist()
        

    site_target_dict.keys()
    return site_target_dict

site_target_dict = get_site_target_dict(foot_pos_dict)

# %%
# read contact.json to build ik_target
def get_contact_array(PLOT = True):
    with open(args.contact_json_path) as f:
        contact_json = json.load(f)
        
    contact_json = contact_json['data'][:max_frame]

    contact_dict = {}
    for contact_ in contact_json:
        time_ = contact_[0]
        contact_dict_ = contact_[1]

        for key, value in contact_dict_.items():
            mapped_key = KEY_MAP[key]
            if mapped_key not in contact_dict.keys():
                contact_dict[mapped_key] = []
            contact_dict[mapped_key].append(value)

    for key, value in contact_dict.items():
        contact_dict[key] = np.array(value)
        assert len(contact_dict[key]) == max_frame

    contact_schedule = np.zeros((max_frame,4))
    for id_ in range(4):
        for frame_i in range(max_frame):
            foot_name = smr_info.foot_names[id_]
            contact_schedule[frame_i, id_] = contact_dict[foot_name][frame_i]

    if PLOT:
        plot_quadruped_contact_schedule(*contact_schedule.T) # TODO: Fix plot function and move to smr package 

    return contact_schedule

contact_array = get_contact_array(PLOT=PLOT)


# %%
# Set ik target and Solve
time_stamped_target = TimeStampedTarget(time_array, site_target_dict, contact_array=contact_array)
ik_target_holder = IKTargetHolder(model, data, smr_info).from_time_stamped_target(time_stamped_target)

smr_agent.set_ik_target(ik_target_holder)
smr_agent.naive_retarget(viewer)

# %%
# Directly transfer base movement
qpos_array_NMR = smr_agent.naive_retarget_result.copy()
qpos_array_NMR[:,:3] = base_pos_array
qpos_array_NMR[:,3:7] = base_quat_array

for i in range(len(qpos_array_NMR)):
    data.qpos = qpos_array_NMR[i]
    mujoco.mj_forward(model, data)
    viewer.cam.lookat = qpos_array_NMR[i][:3]
    viewer.render()

# %%
smr_agent.spatial_retarget(reference_states=qpos_array_NMR, viewer=viewer)

# %%
qpos_array_SMR = smr_agent.spatial_retarget_result.copy()

# %%
for qpos in qpos_array_SMR:
    data.qpos = qpos
    mujoco.mj_forward(model, data)
    viewer.cam.lookat = qpos[:3]
    viewer.render()
    
# %%
foot_height_post_scale = 1.3

site_target_dict = {}
for qpos in qpos_array_SMR:
    data.qpos = qpos
    mujoco.mj_forward(model, data)
    for id_ in range(4):
        foot_name = smr_info.foot_names[id_]
        foot_pos = data.site_xpos[smr_info.id.foot_ids[id_]].copy()
        if foot_name not in site_target_dict.keys():
            site_target_dict[foot_name] = []
            
        foot_pos[2] = foot_pos[2] * foot_height_post_scale
        hip_pos = data.site_xpos[smr_info.id.hip_ids[id_]].copy()
        
        hip_to_foot_global = foot_pos - hip_pos
        hip_to_foot_local = np.dot(quat2R(qpos[3:7]).T, hip_to_foot_global)
        foot_local = hip_to_foot_local + smr_info.home_hip_xpos[id_]
        site_target_dict[foot_name].append(foot_local)
        
        hip_name = smr_info.hip_names[id_]
        if hip_name not in site_target_dict.keys():
            site_target_dict[hip_name] = []
        site_target_dict[hip_name].append(smr_info.home_hip_xpos[id_])

# %%
# Set ik target and Solve
time_stamped_target = TimeStampedTarget(time_array, site_target_dict, contact_array=contact_array)
ik_target_holder = IKTargetHolder(model, data, smr_info).from_time_stamped_target(time_stamped_target)

smr_agent.set_ik_target(ik_target_holder)
smr_agent.naive_retarget(viewer)

# %%
qpos_array_NMR = smr_agent.naive_retarget_result.copy()
qpos_array_NMR[:,:7] = qpos_array_SMR[:,:7]

for i in range(len(qpos_array_NMR)):
    data.qpos = qpos_array_NMR[i]
    mujoco.mj_forward(model, data)
    viewer.cam.lookat = qpos_array_NMR[i][:3]
    viewer.render()

# %%
smr_agent.spatial_retarget(reference_states=qpos_array_NMR, viewer=viewer)

# %%
qpos_array_lifted_foot = smr_agent.spatial_retarget_result.copy()
for qpos in qpos_array_lifted_foot:
    data.qpos = qpos
    mujoco.mj_forward(model, data)
    viewer.cam.lookat = qpos[:3]
    
    for _ in range(3):
        viewer.render()
# %%
for qpos in qpos_array_SMR.copy():
    data.qpos = qpos
    mujoco.mj_forward(model, data)
    viewer.cam.lookat = qpos[:3]
    
    for _ in range(3):
        viewer.render()
# %%
