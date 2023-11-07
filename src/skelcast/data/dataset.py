import numpy as np

from .prepare_data import get_skeleton_files, get_missing_files, filter_missing


def read_skeleton_file(
    file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True
) -> dict:
    """
    Copied from https://github.com/shahroudy/NTURGB-D/blob/master/Python/txt2npy.py
    """
    f = open(file_path, "r")
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat["file_name"] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat["nbodys"] = []
    bodymat["njoints"] = njoints
    for body in range(max_body):
        if save_skelxyz:
            bodymat["skel_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat["rgb_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat["depth_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            continue
        # skip the empty frame
        bodymat["nbodys"].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = "skel_body{}".format(body)
            rgb_body = "rgb_body{}".format(body)
            depth_body = "depth_body{}".format(body)

            bodyinfo = datas[cursor][:-1].split(" ")
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(" ")
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame, joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame, joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame, joint] = jointinfo[5:7]
    # prune the abundant bodys
    try:
        for each in range(max_body):
            if each >= max(bodymat["nbodys"]):
                if save_skelxyz:
                    del bodymat["skel_body{}".format(each)]
                if save_rgbxy:
                    del bodymat["rgb_body{}".format(each)]
                if save_depthxy:
                    del bodymat["depth_body{}".format(each)]
    except ValueError:
        print(f"Error found in file: {file_path}")
    return bodymat
