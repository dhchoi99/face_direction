import numpy as np
import open3d as o3d
import time

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

def pcl_img(img, landmarks=None, project_landmarks=False, uniform_color=False):
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
    pcl_img.colors = o3d.utility.Vector3dVector(colors / 255.)

    if uniform_color:
        pcl_img.paint_uniform_color((0.9, 0.9, 0.9))

    return pcl_img


def show_3d(axes, img, landmarks):
    assert landmarks.shape[1] == 3
    p_lands = pcl_landmarks(landmarks)

    center = landmarks[33]
    p_axes = pcl_axes(axes, center)

    p_img = pcl_img(img, landmarks, True)
    o3d.visualization.draw_geometries([p_lands, p_axes, p_img])




if __name__ == '__main__':
    landmarks = np.random.rand(68, 3) # random landamrks
    landmarks *= 50 # scaling
    landmarks -= landmarks[33]  # translation to (0, 0, 0)
    landmarks += (128, 128, 0) # translation to center of image
    p_lands = pcl_landmarks(landmarks)

    img = np.zeros((256, 256, 3)) # image plane (H, W, C)
    p_img = pcl_img(img, uniform_color=True)

    axes = np.identity(3) # x, y, z axes
    p_axes = pcl_axes(axes, center=np.array([128, 128, 0]), )

    o3d.visualization.draw_geometries([p_lands, p_axes, p_img])