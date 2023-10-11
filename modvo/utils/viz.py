import cv2
import numpy as np
import matplotlib.cm as cm

def draw_keypoints(img, kpts, color=(255,255,255)):
    out = img
    for k in kpts:
        if(color == 'random'):
            c = [np.random.randint(0,255) for _ in range(3)]
        else:
            c = color
        out = cv2.circle(out, (int(k[0]), int(k[1])), 3, c, 2)
    return out


def draw_matches(image0, image1, kpts0, kpts1, scores=None):
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
    return out

def save_image(image, path):
    cv2.imwrite(path, image)

def open_image(path):
    return cv2.imread(path)