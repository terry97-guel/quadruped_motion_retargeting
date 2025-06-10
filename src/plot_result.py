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

SAVE = True
# %%
# for _ in range(10):
#     plot_robot(viewer=viewer, model=model, data=data, qpos_array=motion_read.qpos_array, lookat_site_idr=smr_info.id.trunk_site, sphere_site_ids=smr_info.id.foot_ids, PLOT_EVERY=10)

# %%
import cv2
def grab_image(viewer, resize_rate=None,interpolation=cv2.INTER_NEAREST):
   """
      Grab the rendered iamge
   """
   img = np.zeros((viewer.viewport.height,viewer.viewport.width,3),dtype=np.uint8)
   mujoco.mjr_render(viewer.viewport,viewer.scn,viewer.ctx)
   mujoco.mjr_readPixels(img, None,viewer.viewport,viewer.ctx)
   img = np.flipud(img) # flip image
   # Resize
   if resize_rate is not None:
      h = int(img.shape[0]*resize_rate)
      w = int(img.shape[1]*resize_rate)
      img = cv2.resize(img,(w,h),interpolation=interpolation)
   return img

# %%
lookat_site_idr = smr_info.id.trunk_site
viewer.cam.distance = 2.0
viewer.cam.azimuth = 90
viewer.cam.elevation = -2

from pathlib import Path
save_name = f"../output/{cfg.ROBOT}/{cfg.MOTION}"
save_folder = Path(save_name)
save_folder.mkdir(parents=True, exist_ok=True)

data.mocap_pos[:] = 10

for i in range(0, len(motion_read.qpos_array), 10):
    qpos = motion_read.qpos_array[i]
    mocap_pos = motion_read.mpos_array[i].reshape(-1, 3)
    data.mocap_pos[:] = mocap_pos
    data.qpos[:] = qpos

    mujoco.mj_forward(model, data)
    viewer.cam.lookat = data.site_xpos[lookat_site_idr].copy()
    viewer.render()

    if SAVE:
        img = grab_image(viewer)

        # plot with matplotlib
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        # remove white space
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        # save image
        plt.savefig(save_folder/f"{i:04d}.png")
        plt.close()
    
# # %%
# for idx in range(len(motion_read.qpos_array[::3])):
#     qpos = motion_read.qpos_array[idx*3]
#     mocap_pos = motion_read.mpos_array[idx*3].reshape(-1, 3)

#     data.qpos[:] = qpos
#     data.mocap_pos[:] = mocap_pos
#     mujoco.mj_forward(model, data)
#     viewer.cam.lookat = data.site_xpos[lookat_site_idr].copy()
#     viewer.render()

# %%
# make a video from the images
# import cv2
# image_folder = save_folder
# video_name = f"{save_name}.mp4"
# images = [img for img in sorted(image_folder.glob("*.png"))]
# frame = cv2.imread(str(images[0]))
# height, width, layers = frame.shape
# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
# video.release()

# %%
import cv2
from pathlib import Path

save_name = f"{cfg.ROBOT}_{cfg.MOTION}"  # just a name, no folders here

image_folder = save_folder
video_path = save_folder.parent / f"{save_name}.mp4"

# Ensure the folder exists
image_folder.mkdir(parents=True, exist_ok=True)

# Get image list
images = sorted(image_folder.glob("*.png"))
if not images:
    raise ValueError("No .png images found in folder:", image_folder)

# Load first frame
first_frame = cv2.imread(str(images[0]))
height, width, layers = first_frame.shape

# Use a more compatible codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(str(video_path), fourcc, 25, (width, height))

if not video.isOpened():
    raise RuntimeError(f"Failed to open video writer for {video_path}")

# Write frames
for img_path in images:
    frame = cv2.imread(str(img_path))
    if frame is not None:
        video.write(frame)

video.release()
print(f"Saved video to {video_path.resolve()}")

# %%