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


def should_blacklist(file_path: str):
    with open(file_path, "r") as file:
        try:
            # Read the first line to get the number of following lines
            num_lines = int(file.readline().strip())

            # Read the next 'num_lines' lines
            lines = [file.readline().strip() for _ in range(num_lines)]

            # Check if all the lines are '0' and the count matches num_lines
            if all(line == "0" for line in lines) and len(lines) == num_lines:
                return True
            else:
                return False

        except ValueError:
            # Handle the case where the first line is not a number
            print(f"Error: The file {file_path} does not start with a number.")
            return False
        except Exception as e:
            # Handle other possible exceptions such as file not found, etc.
            print(f"An error occurred: {e}")
            return False
