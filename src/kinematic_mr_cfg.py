from motion_menagerie.mann import MANN_BASE_PATH
from motion_menagerie.mann_lifted import MANN_LIFTED_BASE_PATH
from motion_menagerie import get_MR_json_path
import numpy as np

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


class QuadrupedCfg:
    dataset = "MANN"
    MOTION = "D1_007_KAN01_001" 
    # D1_007_KAN01_001
    # D1_009_KAN01_001
    # D1_009_KAN01_002
    # D1_010_KAN01_002
    # D1_010_KAN01_003
    # D1_010_KAN01_004
    
    # D1_025_KAN01_001
    # D1_047z_KAN01_005
    # D1_049_KAN01_001
    KEY_MAP = KEY_MAP

    if dataset == "MANN":
        MOTION_BASE_PATH = MANN_BASE_PATH
    elif dataset == "MANN_LIFTED":
        MOTION_BASE_PATH = MANN_LIFTED_BASE_PATH
    else:
        raise ValueError("Invalid dataset")
    
    contact_json_path = MOTION_BASE_PATH/"data"/f"{MOTION}_contact.json"
    traj_json_path = MOTION_BASE_PATH/"data"/f"{MOTION}_traj.json"
    
    fps = 60
    dt = 1/fps
    max_frame = -1

    PLOT = True
    foot_offset = np.array([-0.04, 0, 0])
    
    foot_height_post_scale = 1.3
        
    foot_info_dict = {
        "fr": {"site_name": "FR_foot_site", "geom_names": ["FR_foot"]},
        "fl": {"site_name": "FL_foot_site", "geom_names": ["FL_foot"]},
        "rr": {"site_name": "RR_foot_site", "geom_names": ["RR_foot"]},
        "rl": {"site_name": "RL_foot_site", "geom_names": ["RL_foot"]},
    }


    crl_joint_orders = ["trunk",
                "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]

    

    MR = "SMR"
    
class Go2Cfg(QuadrupedCfg):
    ROBOT = "go2_task"
    scale = np.array([0.8,1.0,0.90])
    
class A1Cfg(QuadrupedCfg):
    ROBOT = "a1_task"
    scale = np.array([0.8,1,0.8])
    
cfg = Go2Cfg