import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


def main(img_path, bin_path, calib_path):
    name = str(os.path.split(img_path)[-1]).split('.')[0]
    with open(calib_path, 'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    # read raw data from binary
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    cam = P2 * R0_rect * Tr_velo_to_cam * velo
    cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection staff
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    png = mpimg.imread(img_path)
    IMG_H, IMG_W, _ = png.shape
    # restrict canvas in range
    plt.axis([0, IMG_W, IMG_H, 0])
    plt.imshow(png)
    # filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    # generate color map from depth
    u, v, z = cam
    plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    plt.title(name)
    plt.savefig(f'./{name}.png', bbox_inches='tight')


if __name__ == '__main__':
    calib_base_path = r'C:\Users\NailinLiao\Desktop\kitti-velo2cam-master\testing\calib'
    imge_base_path = r'C:\Users\NailinLiao\Desktop\kitti-velo2cam-master\data_object_image_2\testing\image_2'
    velodyne_base_path = r'C:\Users\NailinLiao\Desktop\kitti-velo2cam-master\data_object_velodyne\testing\velodyne'

    img_list = os.listdir(imge_base_path)
    calib_list = os.listdir(calib_base_path)
    velodyne_list = os.listdir(velodyne_base_path)
    for img_name in img_list:
        # if img_name in calib_list and img_name in velodyne_list:
        print(img_name)
        new_img_name = str(img_name).split('.')[0]
        one_img = os.path.join(imge_base_path, new_img_name + '.png')
        one_calib = os.path.join(calib_base_path, new_img_name + '.txt')
        one_ivelodyne = os.path.join(velodyne_base_path, new_img_name + '.bin')
        # img_path, bin_path, calib_path/
        try:
            main(one_img, one_ivelodyne, one_calib)
        except:
            print('erro', img_name)
