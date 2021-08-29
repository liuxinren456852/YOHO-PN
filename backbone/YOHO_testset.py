import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import sys
import glob
from tqdm import tqdm

import common as cm
from tools import get_pcd, get_keypts
from sklearn.neighbors import KDTree
import importlib
import open3d
from utils.utils import transform_points,PCA_value
from utils.dataset import get_dataset_name


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def build_patch_input(pcd, keypts, vicinity=0.3, num_points_per_patch=1024):
    refer_pts = keypts.astype(np.float32)
    pts = np.array(pcd.points).astype(np.float32)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)
    local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
    for i in range(num_patches):
        local_neighbors = pts[ind_local[i], :]
        if local_neighbors.shape[0] >= num_points_per_patch:
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
        else:
            fix_idx = np.asarray(range(local_neighbors.shape[0]))
            while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
            random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
                                          replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
            local_neighbors = local_neighbors[choice_idx]
            local_neighbors[-1, :] = refer_pts[i, :]
        local_patches[i] = local_neighbors

    return local_patches


def prepare_patch(dataset, pc_id, trans_matrix,r):
    pcd = dataset.get_pc_o3d(pc_id)
    keypts=dataset.get_kps(pc_id).astype(np.float32)
    if is_rotate_dataset:
        # Add arbitrary rotation
        # rotate terminal frament with an arbitrary angle around the z-axis
        angles_3d = np.random.rand(3) * np.pi * 2
        R = cm.angles2rotation_matrix(angles_3d)
        T = np.identity(4)
        T[:3, :3] = R
        pcd.transform(T)
        keypts_pcd = make_open3d_point_cloud(keypts)
        keypts_pcd.transform(T)
        keypts = np.array(keypts_pcd.points)
        trans_matrix.append(T)

    local_patches = build_patch_input(pcd, keypts,vicinity=r, num_points_per_patch=1024)  # [num_keypts, 1024, 4]
    return local_patches


def generate_descriptor(model, desc_name, dataset, descpath,r):
    rotations=np.load('../group_related/Rotation.npy')#60*3*3
    make_open3d_point_cloud(f'{descpath}/pca')
    model.eval()
    num_frag = len(dataset.pc_ids)
    num_desc = len(os.listdir(descpath))
    trans_matrix = []
    if num_frag == num_desc:
        print("Descriptor already prepared.")
        return
    for j in range(num_frag):
        #if os.path.exists(descpath + f"/{j}.npy"):continue
        local_patches = prepare_patch(dataset, j, trans_matrix,r)

        #pca_values
        pcas=[]
        for p_id in range(local_patches.shape[0]):
            patch=local_patches[p_id] #1024*3
            patch=patch-patch[-1,0:3][None,:] #centralize
            pca_one=PCA_value(patch)
            pcas.append(pca_one[None,:])# 1*3
        pcas=np.concatenate(pcas,axis=0)# 5000*3
        np.save(f'{descpath}/pca/{j}.npy',pcas)

        input_=[]
        for p_id in range(local_patches.shape[0]):
            for r_id in range(rotations.shape[0]):
                R=rotations[r_id]
                p=local_patches[p_id][:,0:3]#1024*3
                p=transform_points(p,R)
                input_.append(p[None,:,:])
        input_=np.concatenate(input_,axis=0)

        input_ = torch.tensor(input_.astype(np.float32))
        B = input_.shape[0]
        input_ = input_.cuda()
        model = model.cuda()
        model = model.eval()
        # calculate descriptors
        desc_list = []
        start_time = time.time()
        desc_len = 32
        step_size = 1000 #can judge if out of memory->500
        iter_num = np.int(np.ceil(B / step_size))
        for k in tqdm(range(iter_num)):
            if k == iter_num - 1:
                with torch.no_grad():
                    desc = model(input_[k * step_size:, :, :])
            else:
                with torch.no_grad():
                    desc = model(input_[k * step_size: (k + 1) * step_size, :, :])
            desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
            del desc
        step_time = time.time() - start_time
        print(f'Finish {B} descriptors of {j}-th cloud spend {step_time:.4f}s')
        desc = np.concatenate(desc_list, 0).reshape([-1, 60, desc_len])
        desc = np.swapaxes(desc,1,2)
        np.save(descpath + f"/{j}.npy", desc.astype(np.float32))
    if is_rotate_dataset:
        scene_name = pcdpath.split('/')[-2]
        all_trans_matrix[scene_name] = trans_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ETH',
        type=str,
        help='Dataset name')
    parser.add_argument(
        '--r',
        default=1,
        type=float,
        help='patch radius')
    args = parser.parse_args()
    datasets=get_dataset_name(args.dataset,'../data/origin_data')
    r=args.r

    experiment_id = time.strftime('%m%d%H%M')

    # dynamically load the model
    module_file_path = '../model.py'
    shutil.copy2(os.path.join('.', '../../network/PointNet.py'), module_file_path)
    module_name = ''
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.PN(r)
    model = nn.DataParallel(model, device_ids=[0])
    model_fn = './pretrained_backbone/3DMatch_best.pkl'
    model.load_state_dict(torch.load(model_fn))# pn model pkl list

    all_trans_matrix = {}
    is_rotate_dataset = False
    for scene,dataset in datasets.items():
        descpath = f'../data/YOHO_PN/Testset/{dataset.name}/PN_Input_Group_feature'#YOHO-PN dir  YOHO-PN/Testset/3dmatch/scene
        if not os.path.exists(descpath):
            os.makedirs(descpath)
        start_time = time.time()
        print(f"Begin Processing {scene}")
        generate_descriptor(model, desc_name='PointNet', dataset, descpath=descpath,r=r)
        print(f"Finish in {time.time() - start_time}s")
    if is_rotate_dataset:
        np.save(f"trans_matrix", all_trans_matrix)
