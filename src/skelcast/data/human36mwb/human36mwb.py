import copy

import numpy as np

from skelcast.data.human36mwb.camera import normalize_screen_coordinates
from skelcast.data.human36mwb.human36mwbparams import h36m_cameras_intrinsic_params, h36m_cameras_extrinsic_params
from skelcast.data.human36mwb.skeleton import Skeleton


class MocapDataset:
    def __init__(self, fps, skeleton):
        self._skeleton = skeleton
        self._fps = fps
        self._data = None # Must be filled by subclass
        self._cameras = None # Must be filled by subclass
    
    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                if 'positions' in s:
                    s['positions'] = s['positions'][:, kept_joints]
                
        
    def __getitem__(self, key):
        return self._data[key]
        
    def subjects(self):
        return self._data.keys()
    
    def fps(self):
        return self._fps
    
    def skeleton(self):
        return self._skeleton
        
    def cameras(self):
        return self._cameras
    
    def supports_semi_supervised(self):
        # This method can be overridden
        return False


class Human3WBDataset(MocapDataset):
    def __init__(self, path, seq_len=27):

        self.seq_len = seq_len
        # Load serialized dataset
        check_data = np.load(path, allow_pickle=True)
        self.metadata = check_data['metadata'].item()
        train_data = check_data['train_data'].item()

        # prepare skeleton
        joints_left = self.metadata['left_side']
        joints_right = self.metadata['right_side']

        self.kps_order = ['body', 'left_foot', 'right_foot', 'face', 'left_hand', 'right_hand']
        # below orj parents
        body_parents = [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
        left_foot_parents = [15, 15, 15]
        right_foot_parents = [16, 16, 16]
        face_parents = [0] * len(self.metadata["face"])  # for face, you can use nose as parents of all
        left_hand_parents = [9, 91, 92, 93, 94, 91, 96, 97, 98, 91, 100, 101, 102, 91, 104, 105, 106, 91, 108, 109, 110]
        right_hand_parents = [10, 112, 113, 114, 115, 112, 117, 118, 119, 112, 121, 122, 123, 112, 125, 126, 127, 112, 129, 130, 131]

        self.parents = body_parents + left_foot_parents + right_foot_parents + face_parents + left_hand_parents + right_hand_parents

        self.num_kps = len(self.parents)

        self.keypoints_metadata = {'layout_name': 'h3wb',
                                   'num_joints': self.num_kps,
                                   'keypoints_symmetry': [joints_left, joints_right]
                                   }

        h3wb_skeleton = Skeleton(parents=self.parents,
                                 joints_left=joints_left,
                                 joints_right=joints_right)

        super().__init__(fps=50, skeleton=h3wb_skeleton)

        # we can use the same camera parameters since the intrinsics are the same!!
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
                    'float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        self.camera_order_id = ['54138969', '55011271', '58860488', '60457274']
        self._data = {}
        self._cameras_full_data = {}
        for subject, actions in train_data.items():
            self._data[subject] = {}
            self._cameras_full_data[subject] = [self.metadata[subject][cam] for cam in self.camera_order_id]
            for action_name, act_data in actions.items():
                if len(act_data['global_3d']) < self.seq_len:
                    # if the sequence length is too short lets skip it!
                    continue
                self._data[subject][action_name] = {
                    'positions': act_data['global_3d'].squeeze(),  # global coord
                    'cameras': [self.metadata[subject][cam] for cam in self.camera_order_id],
                    'positions_3d': [act_data[cam]['camera_3d'].squeeze() for cam in self.camera_order_id],
                    'pose_2d': [act_data[cam]['pose_2d'].squeeze() for cam in self.camera_order_id]
                }
