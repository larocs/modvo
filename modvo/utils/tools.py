import numpy as np

def convert_kpts_cv_to_numpy(kpts_cv):
    kpts = np.zeros((len(kpts_cv), 2))
    for i, kpt in enumerate(kpts_cv):
            kpts[i, 0] = kpt.pt[0]
            kpts[i, 1] = kpt.pt[1]
    return kpts

def get_index(nparray, item):
    for idx, val in enumerate(nparray):
        if((val == item).all()):
            return idx
    return None