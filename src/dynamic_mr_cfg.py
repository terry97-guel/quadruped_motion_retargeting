from motion_menagerie.mann import MANN_BASE_PATH
from motion_menagerie.mann_lifted import MANN_LIFTED_BASE_PATH
from motion_menagerie.stmr import STMR_BASE_PATH
from motion_menagerie import get_MR_info_path
import numpy as np


class QuadrupedCfg:
    dataset = "STMR"
    MOTION = "backflip3_long"
    SAVE = False

    ROBOT = "go2box3_task"
    # ROBOT = "go1_task"
    # ROBOT = "go2_task"
    ROBOT_CLASS = "Quadruped"
    # MOTION = "D1_007_KAN01_001"

    if dataset == "MANN":
        MOTION_BASE_PATH = MANN_BASE_PATH
    elif dataset == "MANN_LIFTED":
        MOTION_BASE_PATH = MANN_LIFTED_BASE_PATH
    elif dataset == "STMR":
        MOTION_BASE_PATH = STMR_BASE_PATH
    else:
        raise ValueError("Invalid dataset")
    
    MR = "MJPC"

    foot_info_dict = {
        "fr": {"site_name": "FR_foot_site", "geom_names": ["FR_foot"]},
        "fl": {"site_name": "FL_foot_site", "geom_names": ["FL_foot"]},
        "rr": {"site_name": "RR_foot_site", "geom_names": ["RR_foot"]},
        "rl": {"site_name": "RL_foot_site", "geom_names": ["RL_foot"]},
    }


    crl_joint_orders = ["trunk",
                "FL_hip_joint", "FR_hip_joint", "RR_hip_joint", "RL_hip_joint",
                "FL_thigh_joint", "FR_thigh_joint", "RR_thigh_joint", "RL_thigh_joint",
                "FL_calf_joint", "FR_calf_joint", "RR_calf_joint", "RL_calf_joint"]

    motion_info_from_path = get_MR_info_path(MOTION_BASE_PATH, ROBOT, "SMR").resolve()

cfg = QuadrupedCfg