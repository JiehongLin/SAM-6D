import os
import errno
import shutil
import numpy as np
import json
import sys

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


def write_txt(path, list_files):
    with open(path, "w") as f:
        for idx in list_files:
            f.write(idx + "\n")
        f.close()


def open_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_json(path):
    with open(path, "r") as f:
        # info = yaml.load(f, Loader=yaml.CLoader)
        info = json.load(f)
    return info


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def save_json_bop23(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f)


def save_npz(path, info):
    np.savez_compressed(path, **info)


def casting_format_to_save_json(data):
    # casting for every keys in dict to list so that it can be saved as json
    for key in data.keys():
        if (
            isinstance(data[key][0], np.ndarray)
            or isinstance(data[key][0], np.float32)
            or isinstance(data[key][0], np.float64)
            or isinstance(data[key][0], np.int32)
            or isinstance(data[key][0], np.int64)
        ):
            data[key] = np.array(data[key]).tolist()
    return data


def get_root_project():
    return os.path.dirname(os.path.dirname((os.path.abspath(__file__))))


def append_lib(path):
    sys.path.append(os.path.join(path, "src"))