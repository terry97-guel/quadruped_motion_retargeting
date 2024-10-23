from motion_menagerie.mann import MANN_BASE_PATH
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

class CommonCfg:
    MOTION = "D1_009_KAN01_002" 
    # MOTION = "D1_047z_KAN01_005"
    # MOTION = "D1_009_KAN01_001"
    # MOTION = "D1_010_KAN01_004"
    
    KEY_MAP = KEY_MAP
    
    
    contact_json_path = MANN_BASE_PATH/"data"/f"{MOTION}_contact.json"
    traj_json_path = MANN_BASE_PATH/"data"/f"{MOTION}_traj.json"
    dt = 1/60
    max_frame = -1

    PLOT = True
    foot_offset = np.array([-0.04, 0, 0])
    
    foot_height_post_scale = 1.3

class Go2Cfg(CommonCfg):
    ROBOT = "go2_task"
    scale = np.array([0.8,1.0,0.90])
    
class A1Cfg(CommonCfg):
    ROBOT = "a1_task"
    scale = np.array([0.8,1,0.8])
    
cfg = Go2Cfg