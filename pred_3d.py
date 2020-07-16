from glob import glob
import os
import pickle as pkl
import time

import cv2
import numpy as np
from PIL import Image

import face_alignment
import open3d as o3d


def rule_3d_x(landmarks):
    eps = 1e-10
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.asarray(landmarks, dtype=float)
    assert (68, 3) == landmarks.shape

    x = []
    for i in range(8): # use cheek landmark
        vec = landmarks[16-i] - landmarks[i]
        vec /= (np.linalg.norm(vec)+eps)
        x.append(vec)
    '''
    for i in range(17, 22):
        vec = landmarks[i] - landmarks[43-i]
        vec /= np.linalg.norm(vec)
        x.append(vec)
        '''
    x = np.asarray(x)
    x = np.mean(x, axis=0)
    x /= (np.linalg.norm(x)+eps)
    x = np.asarray(x).reshape(-1)
    return x

def rule_3d_y(landmarks):
    eps = 1e-10
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.asarray(landmarks, dtype=float)
    assert (68, 3) == landmarks.shape

    y = []
    y.append(landmarks[27] - landmarks[57])

    y /= (np.linalg.norm(y) + eps)
    y = np.asarray(y).reshape(-1)
    return y

def axes_3d(landmarks):
    x = rule_3d_x(landmarks)
    y = rule_3d_y(landmarks)
    z = np.cross(x, y)
    axes = x, y, z
    return axes

def benchmark_3d():
    path = 'data/voxceleb'
    paths_label = sorted(glob(os.path.join(path, 'annos', '*.*')))
    paths_img = sorted(glob(os.path.join(path, 'image', '*.*')))

    fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    max = 0
    for i in range(len(paths_label)):
        path_label = paths_label[i]
        path_img = paths_img[i]
        img = Image.open(path_img).convert('RGB')
        img_np = np.asarray(img)
        landmarks = fa3d.get_landmarks_from_image(img_np)
        landmarks = landmarks[0]
        '''#with open(path_label, 'rb') as f:
            landmarks = pkl.load(f)
            f.close()
        landmarks = pkl.load(path_label)
        '''

        x, y, z = axes_3d(landmarks)
        d = np.dot(x, y)
        if np.abs(d) > max:
            max = np.abs(d)

        #axes = (x, y, z)
        #show_3d(axes, img, landmarks)
    print('max', max)


def pcl_landmarks(landmarks):
    pcl_lands = o3d.geometry.PointCloud()
    pcl_lands.points = o3d.utility.Vector3dVector(landmarks)
    return pcl_lands

def pcl_axes(axes, center, align_image=False):
    if align_image:
        center = center.copy()
        center[2] = 0
    x, y, z = axes
    vec = o3d.geometry.PointCloud()
    mult = (np.arange(-100, 100) / 3.).reshape(-1, 1)
    pointsx = center + mult * x
    pointsy = center + mult * y
    pointsz = center + mult * z
    points = np.concatenate((pointsx, pointsy, pointsz), axis=0)
    vec.points = o3d.utility.Vector3dVector(points)
    return vec

def pcl_img(img, landmarks=None, project_landmarks=False):
    assert isinstance(img, np.ndarray)
    colors = img.copy()
    xy = np.indices(colors.shape[:2])
    z = np.zeros((1, xy.shape[1], xy.shape[2]))
    points = np.concatenate((xy, z), axis=0)
    points = points.transpose((2, 1, 0)).reshape(-1, 3)

    if project_landmarks and landmarks is not None:
        proj = landmarks.copy()
        proj[:, 2] = 0
        for i in range(len(proj)):
            point = proj[i]
            x, y = int(point[0]), int(point[1])
            H, W = colors.shape[0], colors.shape[1]
            r = 1
            colors[(y - r) % H:(y + r) % H, (x - r) % W:(x + r) % H, :] = np.array([0, 255, 0])

    colors = colors.reshape(-1, 3)
    pcl_img = o3d.geometry.PointCloud()
    pcl_img.points = o3d.utility.Vector3dVector(points)
    pcl_img.paint_uniform_color((0.9, 0.9, 0.9))
    pcl_img.colors = o3d.utility.Vector3dVector(colors / 255.)

    return pcl_img


def show_3d(axes, img, landmarks):
    assert landmarks.shape[1] == 3
    p_lands = pcl_landmarks(landmarks)

    center = landmarks[33]
    p_axes = pcl_axes(axes, center)

    p_img = pcl_img(img, landmarks, True)
    o3d.visualization.draw_geometries([p_lands, p_axes, p_img])


def direction_3d(path, path_img):
    path = 'data/voxceleb/annos/0000098.pkl'
    path_img = 'data/voxceleb/image/0000098.png'
    landmarks = np.load(path, allow_pickle=True)

    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)

    x = rule_3d_x(landmarks)
    y = rule_3d_y(landmarks)
    print('xy dot', np.dot(x, y))
    print('xy dot as angle', np.arccos(np.dot(x, y)) / np.pi * 180. )
    z = np.cross(x, y)

    axes = (x, y, z)

    show_3d(axes, img, landmarks)


def main():
    start_time = time.time()
    #direction_3d(None, None)
    benchmark_3d()
    print('time', time.time()-start_time)


if __name__ == '__main__':
    main()