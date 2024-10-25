# %%
from mjtools import reset, plot_robot
from smr.task.Quadruped.info import QuadrupedSMRInfo

from robot_menagerie import ASSET_XML_DICT

from smr.agent import SMR
from smr.ik_target_holder import IKTargetHolder

import numpy as np
import time
import mujoco
import mujoco_viewer
from matplotlib import pyplot as plt
import json

# from kinematic_mr_cfg import cfg
from dynamic_mr_cfg import cfg

xml_path = ASSET_XML_DICT[cfg.ROBOT]
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
from smr.ik_target_holder import IKTargetHolder, TimeStampedTarget
smr_info  = QuadrupedSMRInfo(model, data, only_foot=True)

# %%

from motion_menagerie import MotionIO
motion_xml_path = cfg.MOTION_BASE_PATH/f"{cfg.MR}/{cfg.ROBOT}/xml/{cfg.MOTION}.xml"
motion_read = MotionIO(model, data, viewer).read_motion_xml(motion_xml_path)

# %%
plot_robot(viewer=viewer, model=model, data=data, qpos_array=motion_read.qpos_array, lookat_site_idr=smr_info.id.trunk_site, sphere_site_ids=smr_info.id.foot_ids, PLOT_EVERY=10)

# %%
