from glob import glob
import os
import time

import cv2
import numpy as np

LEN_LANDMARK = 68


def find_x_3d(source_im):
    '''
    find x axes from given landmarks
    source_im: landmarks, (n * 68 * 3)
    returns: x axes, (n * 3)
    '''
    eps = 1e-10

    if not isinstance(source_im, np.ndarray):
        source_im = np.asarray(source_im, dtype=float)
    assert (LEN_LANDMARK, 3) == source_im.shape[-2:]

    xs = source_im[:, 0:8:1, :] - source_im[:, 16:8:-1, :]
    xs /= np.linalg.norm(xs, axis=2, keepdims=True) + eps
    x = np.mean(xs, axis=1)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + eps
    x = x.reshape(-1, 3)

    return x


def find_y_3d(source_im):
    '''
    find y axes from given landmarks
    source_im: landmarks, (n * 68 * 3)
    returns: y axes, (n * 3)
    '''
    eps = 1e-10

    if not isinstance(source_im, np.ndarray):
        source_im = np.asarray(source_im, dtype=float)
    assert (LEN_LANDMARK, 3) == source_im.shape[-2:]

    ys = source_im[:, 27, :] - source_im[:, 33, :]
    y = ys / (np.linalg.norm(ys, axis=1, keepdims=True) + eps)

    return y


def find_frontal_face_3d(source_im):
    '''
    source_im: landmarks, (n * 68 * 3)
    returns: scores, (n, )
    '''
    xs = find_x_3d(source_im)
    ys = find_y_3d(source_im)
    zs = np.cross(xs, ys)
    scores = zs[:, 2]
    return scores


def find_frontal_face_nose(source_im):
    '''
    source_im: landmarks, (n * 68 * 2)
    returns: scores, (n, )
    '''
    eps = 1e-10

    if not isinstance(source_im, np.ndarray):
        source_im = np.asarray(source_im)
    assert (LEN_LANDMARK, 2) == source_im.shape[-2:]

    vec1 = source_im[:, 30, :] - source_im[:, 27, :]
    vec1 /= np.linalg.norm(vec1, axis=1, keepdims=True) + eps
    vec2 = source_im[:, 33, :] - source_im[:, 30, :]
    vec2 /= np.linalg.norm(vec2, axis=1, keepdims=True) + eps

    angle = np.sum(np.multiply(vec1, vec2), axis=1)
    scores = angle
    # angle = np.arccos(angle) * 180. / np.pi
    # scores = 180 - angle

    dist = source_im[:, 27, :] - source_im[:, 33, :]
    dist = np.linalg.norm(dist, axis=1)

    scores = scores + 0.0001 * dist

    return scores


if __name__ == '__main__':
    path = 'data/voxceleb/annos'
    paths = sorted(glob(os.path.join(path, '*.*')))
    inps = []
    for path in paths:
        inp = np.load(path, allow_pickle=True)
        inps.append(inp)
    inps = np.asarray(inps)
    print(inps.shape)
    inps = inps[..., :2]
    scores = find_frontal_face_nose(inps)
    print(scores.shape)
    # print(scores)
    print(np.sort(scores)[::-1])
    print(np.argsort(scores)[::-1])
