import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
# TODO: return the rotation (in radian)

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1.]])

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1.]])

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1.]])


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def pose_spherical2(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    rot_mx = np.stack([rot_theta(x/180. * np.pi) for x in theta], 0)
    c2w = np.einsum('bij, jk -> bik', rot_mx, c2w)
    c2w = np.einsum('ij, bjk -> bik', np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]), c2w)
    return c2w

def load_blender_data(basedir, half_res=False, testskip=1, sim=True):
    splits = ['train', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir+f'_{s}', 'transforms_new.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_rots = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        rots = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames']:
            fname = os.path.join(basedir+f'_{s}', frame['file_name'] + '.png')
            imgs.append(imageio.imread(fname))
            rots.append(float(frame['rotation']))
            if sim:
                poses.append(np.array(frame['simulated_matrix']))
            else:
                poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        if max(rots) == min(rots):
            all_rots.append(np.cumsum(np.array(rots)) - rots[0])
        else:
            all_rots.append(np.array(rots))
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    rots = np.concatenate(all_rots, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_poses = pose_spherical2(np.linspace(-180,180,40+1)[:-1], -30.0, 4.0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

        
    return imgs, poses, render_poses, [H, W, focal], i_split, rots


