from enum import IntEnum


class KinectSkeleton(IntEnum):
    """
    Enum that represents the Kinect's skeleton joints and their indices.
    """
    SPINEBASE = 0
    SPINEMID = 1
    NECK = 2
    HEAD = 3
    SHOULDERLEFT = 4
    ELBOWLEFT = 5
    WRISTLEFT = 6
    HANDLEFT = 7
    SHOULDERRIGHT = 8
    ELBOWRIGHT = 9
    WRISTRIGHT = 10
    HANDRIGHT = 11
    HIPLEFT = 12
    KNEELEFT = 13
    ANKLELEFT = 14
    FOOTLEFT = 15
    HIPRIGHT = 16
    KNEERIGHT = 17
    ANKLERIGHT = 18
    FOOTRIGHT = 19
    SPINESHOULDER = 20
    HANDTIPLEFT = 21
    THUMBLEFT = 22
    HANDTIPRIGHT = 23
    THUMBRIGHT = 24

    def connections():
        """
        Returns a list of tuples that represent the connections between joints.

        Connections:
        ---

        >>> (0, 1),  # SPINEBASE to SPINEMID
        >>> (1, 20), # SPINEMID to SPINESHOULDER
        >>> (20, 2), # SPINESHOULDER to NECK
        >>> (2, 3),  # NECK to HEAD
        >>> (20, 4), # SPINESHOULDER to SHOULDERLEFT
        >>> (4, 5),  # SHOULDERLEFT to ELBOWLEFT
        >>> (5, 6),  # ELBOWLEFT to WRISTLEFT
        >>> (6, 7),  # WRISTLEFT to HANDLEFT
        >>> (7, 22), # HANDLEFT to THUMBLEFT
        >>> (7, 21), # HANDLEFT to HANDTIPLEFT
        >>> (20, 8), # SPINESHOULDER to SHOULDERRIGHT
        >>> (8, 9),  # SHOULDERRIGHT to ELBOWRIGHT
        >>> (9, 10), # ELBOWRIGHT to WRISTRIGHT
        >>> (10, 11),# WRISTRIGHT to HANDRIGHT
        >>> (11, 24),# HANDRIGHT to THUMBRIGHT
        >>> (11, 23),# HANDRIGHT to HANDTIPRIGHT
        >>> (0, 12), # SPINEBASE to HIPLEFT
        >>> (12, 13),# HIPLEFT to KNEELEFT
        >>> (13, 14),# KNEELEFT to ANKLELEFT
        >>> (14, 15),# ANKLELEFT to FOOTLEFT
        >>> (0, 16), # SPINEBASE to HIPRIGHT
        >>> (16, 17),# HIPRIGHT to KNEERIGHT
        >>> (17, 18),# KNEERIGHT to ANKLERIGHT
        >>> (18, 19),# ANKLERIGHT to FOOTRIGHT

        Returns:
        ---

        - connections (list): A list of tuples that represent the connections between joints.
        """
        return [
            (KinectSkeleton.SPINEBASE, KinectSkeleton.SPINEMID),
            (KinectSkeleton.SPINEMID, KinectSkeleton.SPINESHOULDER),
            (KinectSkeleton.SPINESHOULDER, KinectSkeleton.NECK),
            (KinectSkeleton.NECK, KinectSkeleton.HEAD),
            (KinectSkeleton.SPINESHOULDER, KinectSkeleton.SHOULDERLEFT),
            (KinectSkeleton.SHOULDERLEFT, KinectSkeleton.ELBOWLEFT),
            (KinectSkeleton.ELBOWLEFT, KinectSkeleton.WRISTLEFT),
            (KinectSkeleton.WRISTLEFT, KinectSkeleton.HANDLEFT),
            (KinectSkeleton.HANDLEFT, KinectSkeleton.THUMBLEFT),
            (KinectSkeleton.HANDLEFT, KinectSkeleton.HANDTIPLEFT),
            (KinectSkeleton.SPINESHOULDER, KinectSkeleton.SHOULDERRIGHT),
            (KinectSkeleton.SHOULDERRIGHT, KinectSkeleton.ELBOWRIGHT),
            (KinectSkeleton.ELBOWRIGHT, KinectSkeleton.WRISTRIGHT),
            (KinectSkeleton.WRISTRIGHT, KinectSkeleton.HANDRIGHT),
            (KinectSkeleton.HANDRIGHT, KinectSkeleton.THUMBRIGHT),
            (KinectSkeleton.HANDRIGHT, KinectSkeleton.HANDTIPRIGHT),
            (KinectSkeleton.SPINEBASE, KinectSkeleton.HIPLEFT),
            (KinectSkeleton.HIPLEFT, KinectSkeleton.KNEELEFT),
            (KinectSkeleton.KNEELEFT, KinectSkeleton.ANKLELEFT),
            (KinectSkeleton.ANKLELEFT, KinectSkeleton.FOOTLEFT),
            (KinectSkeleton.SPINEBASE, KinectSkeleton.HIPRIGHT),
            (KinectSkeleton.HIPRIGHT, KinectSkeleton.KNEERIGHT),
            (KinectSkeleton.KNEERIGHT, KinectSkeleton.ANKLERIGHT),
            (KinectSkeleton.ANKLERIGHT, KinectSkeleton.FOOTRIGHT),
        ]

    def parent_scheme():
        return [
    0,  # SPINE_BASE (root, its own parent)
    0,  # SPINE_MID's parent is SPINE_BASE
    1,  # NECK's parent is SPINE_MID
    2,  # HEAD's parent is NECK
    1,  # SHOULDER_LEFT's parent is SPINE_MID
    4,  # ELBOW_LEFT's parent is SHOULDER_LEFT
    5,  # WRIST_LEFT's parent is ELBOW_LEFT
    6,  # HAND_LEFT's parent is WRIST_LEFT
    7,  # HAND_TIP_LEFT's parent is HAND_LEFT
    7,  # THUMB_LEFT's parent is HAND_LEFT
    1,  # SHOULDER_RIGHT's parent is SPINE_MID
    10, # ELBOW_RIGHT's parent is SHOULDER_RIGHT
    11, # WRIST_RIGHT's parent is ELBOW_RIGHT
    12, # HAND_RIGHT's parent is WRIST_RIGHT
    13, # HAND_TIP_RIGHT's parent is HAND_RIGHT
    13, # THUMB_RIGHT's parent is HAND_RIGHT
    0,  # HIP_LEFT's parent is SPINE_BASE
    16, # KNEE_LEFT's parent is HIP_LEFT
    17, # ANKLE_LEFT's parent is KNEE_LEFT
    18, # FOOT_LEFT's parent is ANKLE_LEFT
    0,  # HIP_RIGHT's parent is SPINE_BASE
    20, # KNEE_RIGHT's parent is HIP_RIGHT
    21, # ANKLE_RIGHT's parent is KNEE_RIGHT
    22, # FOOT_RIGHT's parent is ANKLE_RIGHT
    1   # SPINE_SHOULDER's parent is SPINE_MID
]