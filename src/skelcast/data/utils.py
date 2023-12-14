"""Copied from https://github.com/qxcv/pose-prediction/blob/master/expmap.py"""
import torch
import numpy as np


def _toposort_visit(parents, visited, toposorted, joint):
    parent = parents[joint]
    visited[joint] = True
    if parent != joint and not visited[parent]:
        _toposort_visit(parents, visited, toposorted, parent)
    toposorted.append(joint)

def check_toposorted(parents, toposorted):
    # check that array contains all/only joint indices
    assert sorted(toposorted) == list(range(len(parents)))

    # make sure that order is correct
    to_topo_order = {
        joint: topo_order
        for topo_order, joint in enumerate(toposorted)
    }
    for joint in toposorted:
        assert to_topo_order[joint] >= to_topo_order[parents[joint]]

    # verify that we have only one root
    joints = range(len(parents))
    assert sum(parents[joint] == joint for joint in joints) == 1

def toposort(parents):
    """Return toposorted array of joint indices (sorted root-first)."""
    toposorted = []
    visited = np.zeros_like(parents, dtype=bool)
    for joint in range(len(parents)):
        if not visited[joint]:
            _toposort_visit(parents, visited, toposorted, joint)

    check_toposorted(parents, toposorted)

    return np.asarray(toposorted)

def _norm_bvecs(bvecs):
    """Norm bone vectors, handling small magnitudes by zeroing bones."""
    bnorms = np.linalg.norm(bvecs, axis=-1)
    mask_out = bnorms <= 1e-5
    # implicit broadcasting is deprecated (?), so I'm doing this instead
    _, broad_mask = np.broadcast_arrays(bvecs, mask_out[..., None])
    bvecs[broad_mask] = 0
    bnorms[mask_out] = 1
    return bvecs / bnorms[..., None]

def xyz_to_expmap(xyz_seq, parents):
    """Converts a tree of (x, y, z) positions into the parameterisation used in
    the SRNN paper, "modelling human motion with binary latent variables"
    paper, etc. Stores inter-frame offset in root joint position."""
    assert xyz_seq.ndim == 3 and xyz_seq.shape[2] == 3, \
        "Wanted TxJx3 array containing T skeletons, each with J (x, y, z)s"

    exp_seq = np.zeros_like(xyz_seq)
    toposorted = toposort(parents)
    # [1:] ignores the root; apart from that, processing order doesn't actually
    # matter
    for child in toposorted[1:]:
        parent = parents[child]
        bones = xyz_seq[:, parent] - xyz_seq[:, child]
        grandparent = parents[parent]
        if grandparent == parent:
            # we're the root; parent bones will be constant (x,y,z)=(0,-1,0)
            parent_bones = np.zeros_like(bones)
            parent_bones[:, 1] = -1
        else:
            # we actually have a parent bone :)
            parent_bones = xyz_seq[:, grandparent] - xyz_seq[:, parent]

        # normalise parent and child bones
        norm_bones = _norm_bvecs(bones)
        norm_parent_bones = _norm_bvecs(parent_bones)
        # cross product will only be used to get axis around which to rotate
        cross_vecs = np.cross(norm_parent_bones, norm_bones)
        norm_cross_vecs = _norm_bvecs(cross_vecs)
        # dot products give us rotation angle
        cos_values = np.sum(norm_bones * norm_parent_bones, axis=-1)
        clamped_cos_values = np.clip(cos_values, -1.0, 1.0)
        angles = np.arccos(clamped_cos_values)
        log_map = norm_cross_vecs * angles[..., None]
        exp_seq[:, child] = log_map

    # root will store distance from previous frame
    root = toposorted[0]
    exp_seq[1:, root] = xyz_seq[1:, root] - xyz_seq[:-1, root]

    return torch.from_numpy(exp_seq).unsqueeze(1)

def exp_to_rotmat(exp):
    """Convert rotation paramterised as exponential map into ordinary 3x3
    rotation matrix."""
    assert exp.shape == (3, ), "was expecting expmap vector"

    # begin by normalising all exps
    angle = np.linalg.norm(exp)
    if angle < 1e-5:
        # assume no rotation
        return np.eye(3)
    dir = exp / angle

    # Rodrigues' formula, matrix edition
    K = np.array([[0, -dir[2], dir[1]], [dir[2], 0, -dir[0]],
                  [-dir[1], dir[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

def exps_to_quats(exps):
    """Turn tensor of exponential map angles into quaternions. If using with
    {xyz,expmap}_to_{expmap,xyz}, remember to remove root node before using
    this!"""
    # See
    # https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula#Rotation_angle_and_rotation_axis

    # flatten the matrix to save my own sanity (numpy Boolean array indexing is
    # super confusing)
    num_exps = int(np.prod(exps.shape[:-1]))
    assert exps.shape[-1] == 3
    exps_flat = exps.reshape((num_exps, 3))
    rv_flat = np.zeros((num_exps, 4))

    # get angles & set zero-rotation vecs to be zero-rotation quaternions (w=1)
    angles = np.linalg.norm(exps_flat, axis=-1)
    zero_mask = angles < 1e-5
    rv_flat[zero_mask, 0] = 1

    # everthing else gets a meaningful value
    nonzero_mask = ~zero_mask
    nonzero_angles = angles[nonzero_mask]
    nonzero_exps_flat = exps_flat[nonzero_mask, :]
    nonzero_normed = nonzero_exps_flat / nonzero_angles[..., None]
    sines = np.sin(nonzero_angles / 2)
    rv_flat[nonzero_mask, 0] = np.cos(nonzero_angles / 2)
    rv_flat[nonzero_mask, 1:] = nonzero_normed * sines[..., None]

    rv_shape = exps.shape[:-1] + (4, )
    return torch.from_numpy(rv_flat.reshape(rv_shape)).unsqueeze(1)
