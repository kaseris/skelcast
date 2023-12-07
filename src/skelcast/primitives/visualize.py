import time

from enum import Enum
from typing import Union

import numpy as np
import torch
import open3d as o3d

from skelcast.primitives.skeleton import KinectSkeleton


class Colors(Enum):
    """
    Enum that represents the colors used for visualizing the skeleton.
    """
    RED = [1, 0, 0]
    GREEN = [0, 1, 0]
    BLUE = [0, 0, 1]
    YELLOW = [1, 1, 0]
    CYAN = [0, 1, 1]
    MAGENTA = [1, 0, 1]
    WHITE = [1, 1, 1]
    BLACK = [0, 0, 0]



def visualize_skeleton(skeleton: Union[np.ndarray, torch.Tensor],
                       framerate: int = 30,
                       skeleton_type: str = 'kinect'):
    assert isinstance(skeleton, (np.ndarray, torch.Tensor)), f'Expected a numpy array or a PyTorch tensor, got {type(skeleton)} instead.'
    # We assume that the skeleton movement has a shape of (seq_len, n_joints, 3)
    if isinstance(skeleton, torch.Tensor):
        skeleton = skeleton.to(torch.float64).numpy()
    assert len(skeleton.shape) == 3, f'Expected a 3-dimensional array, got {len(skeleton.shape)} dimensions instead.'
    assert skeleton.shape[2] == 3, f'Expected the last dimension to be 3, got {skeleton.shape[2]} instead.'
    seq_len, n_joints, _ = skeleton.shape
    if skeleton_type == 'kinect':
        assert n_joints == 25, f'Expected the second dimension to be 25, got {n_joints} instead.'
        connections = KinectSkeleton.connections()
    
    # Create a point cloud object and a line set object
    # These serve as containers for the skeleton data and the connections between joints
    point_cloud = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for timestep in range(seq_len):
        # Update point cloud for the current timestep
        point_cloud.points = o3d.utility.Vector3dVector(skeleton[timestep])
        point_cloud.paint_uniform_color(Colors.RED.value)  # Red color for joints

        bone_lines = [[i.value, j.value] for i, j in connections]
        line_set.lines = o3d.utility.Vector2iVector(bone_lines)
        line_set.points = o3d.utility.Vector3dVector(skeleton[timestep])
        line_set.colors = o3d.utility.Vector3dVector([Colors.BLUE.value for _ in connections])  # Blue color for connections

        if timestep == 0:
            vis.add_geometry(point_cloud)
            vis.add_geometry(line_set)
        else:
            vis.update_geometry(point_cloud)
            vis.update_geometry(line_set)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(1.0 / framerate)

    vis.destroy_window()
