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
