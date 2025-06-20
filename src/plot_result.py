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
data = mujoco.MjData(model)
reset(model, data)

homepos = data.qpos[:3].copy()
home_site_pos = data.site_xpos.copy()

try:
    viewer.close()
except Exception:
    pass
viewer = mujoco_viewer.MujocoViewer(
    model,
    data,
    mode="window",
    title="MPC",
    width=1920,
    height=1080,
    hide_menus=True,
)

viewer.cam.lookat = data.qpos[:3]
viewer.render()

# %%
from smr.ik_target_holder import IKTargetHolder, TimeStampedTarget

smr_info = QuadrupedSMRInfo(model, data, only_foot=True)

# %%

from motion_menagerie import MotionIO

motion_xml_path = cfg.MOTION_BASE_PATH / f"{cfg.MR}/{cfg.ROBOT}/xml/{cfg.MOTION}.xml"
motion_read = MotionIO(model, data, viewer).read_motion_xml(motion_xml_path)

import json

with open(cfg.motion_info_from_path, "r") as f:
    motion_info = json.load(f)

# %%
# for _ in range(10):
#     plot_robot(viewer=viewer, model=model, data=data, qpos_array=motion_read.qpos_array, lookat_site_idr=smr_info.id.trunk_site, sphere_site_ids=smr_info.id.foot_ids, PLOT_EVERY=10)

# %%
import cv2


def grab_image(viewer, resize_rate=None, interpolation=cv2.INTER_NEAREST):
    """
    Grab the rendered iamge
    """
    img = np.zeros((viewer.viewport.height, viewer.viewport.width, 3), dtype=np.uint8)
    mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
    mujoco.mjr_readPixels(img, None, viewer.viewport, viewer.ctx)
    img = np.flipud(img)  # flip image
    # Resize
    if resize_rate is not None:
        h = int(img.shape[0] * resize_rate)
        w = int(img.shape[1] * resize_rate)
        img = cv2.resize(img, (w, h), interpolation=interpolation)
    return img


# %%
lookat_site_idr = smr_info.id.trunk_site
viewer.cam.distance = 3.7
viewer.cam.azimuth = 90
viewer.cam.elevation = -2

from pathlib import Path

save_name = f"output/{cfg.ROBOT}"
save_folder = Path(save_name)
save_folder.mkdir(parents=True, exist_ok=True)

data.mocap_pos[:] = 10

from tqdm.notebook import tqdm

cam_pos = (
    motion_read.qpos_array[0, :3].copy() + motion_read.qpos_array[-1, :3].copy()
) / 2
viewer.cam.lookat = cam_pos

for i in tqdm(range(0, len(motion_read.qpos_array), 10)):
    qpos = motion_read.qpos_array[i]
    data.qpos[:] = qpos

    mujoco.mj_forward(model, data)
    viewer.render()

    img = grab_image(viewer, resize_rate=1.0)

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
    plt.savefig(save_folder / f"{i:04d}.png")
    plt.close()

# %%
# Save to video
import cv2


def save_video_from_images(image_folder, video_path, fps):
    """
    Save a video from a folder of images.
    """
    images = sorted(Path(image_folder).glob("*.png"))
    if not images:
        print("No images found in the folder.")
        return

    # Get the size of the first image
    first_image = cv2.imread(str(images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image_path in images:
        img = cv2.imread(str(image_path))
        out.write(img)

    out.release()
    print(f"Video saved to {video_path}")


# Save the video
video_path = save_folder.parent / f"{cfg.ROBOT}_{cfg.MOTION}.mp4"
save_video_from_images(save_folder, video_path, fps=50)

# %%
