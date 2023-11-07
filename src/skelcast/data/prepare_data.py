import os
import os.path as osp


def get_missing_files(missing_files_dir):
    missing_skel_files = []
    for fname in os.listdir(missing_files_dir):
        print(fname)
        with open(osp.join(missing_files_dir, fname)) as f:
            for idx, line in enumerate(f):
                if idx > 2:
                    missing_skel_files.append(line)
    return missing_skel_files


def get_skeleton_files(dataset_dir: str):
    skeleton_files = []
    # Walk through the directory
    for root, dirs, files in os.walk(dataset_dir):
        # Find all '.skeleton' files within the current root directory
        for file in files:
            if file.endswith(".skeleton"):
                # Append the full path to the list
                skeleton_files.append(os.path.join(root, file))
    return skeleton_files


def filter_missing(skeleton_files: list, missing_skeleton_names: list):
    filtered_skeleton_files = [
        f
        for f in skeleton_files
        if os.path.splitext(os.path.basename(f))[0] not in missing_skeleton_names
    ]
    print(f"Skeleton files after filtering: {len(filtered_skeleton_files)} files left.")
    return filtered_skeleton_files
