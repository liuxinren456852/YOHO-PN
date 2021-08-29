import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
from tqdm import tqdm
import torch.nn as nn
import sys
import glob

import common as cm
from dataloader import get_dataloader
from tools import get_pcd, get_keypts
from sklearn.neighbors import KDTree
import importlib
import torch.nn.functional as F
import open3d
from utils.utils import transform_points,random_rotation_matrix,save_pickle,make_non_exists_dir
from utils.r_eval import compute_R_diff,quaternion_from_matrix

class Set_creater(object):
    def __init__(self, train_batch_dir,val_batch_dir,batch_size):
        # parameters
        # dataloader
        self.data_train_dir = '../data/3DMatch/patches'
        self.data_val_dir = '../data/3DMatch/patches'
        self.train_batch_dir=train_batch_dir
        self.val_batch_dir=val_batch_dir
        self.batch_size=batch_size  
        self.Rgroup=np.load('../group_related/Rotation.npy')
        self.train_loader = get_dataloader(root=self.data_train_dir,
                                           batch_size=self.batch_size,
                                           split='train',
                                           shuffle=False,
                                           num_workers=0,  # if the dataset is offline generated, must 0
                                           )
        self.val_loader = get_dataloader(root=self.data_val_dir,
                                         batch_size=self.batch_size,
                                         split='val',
                                         shuffle=True,
                                         num_workers=0,  # if the dataset is offline generated, must 0
                                         )

        print("Training set size:", self.train_loader.dataset.__len__())
        print("Validate set size:", self.val_loader.dataset.__len__())
        self.yoho_train_num=int(self.train_loader.dataset.__len__()/self.batch_size)
        print("YOHO training set size:", self.yoho_train_num)

        self.load_model()


    def load_model(self):
        # dynamically load the model
        module_file_path = '../model.py'
        module_name = ''
        module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.model = module.PN(0.30)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.load_state_dict(torch.load('./pretrained_backbone/3DMatch_best.pkl'))# pn model pkl 
        self.model = self.model.cuda()
        self.model = self.model.eval()

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id
    
    def DeltaR(self,R,index):
        R_anchor=self.Rgroup[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)


    def yoho_trainset_create(self,times=0):
        with torch.no_grad():
            for iter, (anc_local_patch, pos_local_patch, rotate, shift) in enumerate(tqdm(self.train_loader)):
                #if os.path.exists(f'{self.train_batch_dir}/{times*self.yoho_train_num+iter}.pth'):
                #    continue
                B = anc_local_patch.shape[0]
                anc_local_patch = anc_local_patch.float().numpy()
                pos_local_patch = pos_local_patch.float().numpy()
                rotate = rotate.float().numpy()
                shift = shift.float().numpy()

                anc_local_patch_g=[]
                pos_local_patch_g=[]
                rotate_g=[]
                index_g=[]
                deltaR_g=[]
                for i in range(B):
                    anc=anc_local_patch[i]#1024*3
                    pos=pos_local_patch[i]#1024*3
                    #alignment
                    R_former=rotate[i]#3*3
                    pos=transform_points(pos,R_former)
                    #random rotate
                    R=random_rotation_matrix(180)
                    ind=self.R2DR_id(R)
                    deltaR=self.DeltaR(R,ind)
                    pos=transform_points(pos,R)#对pos random
                    #batch_append
                    rotate_g.append(R[None,:,:])
                    index_g.append(ind)
                    deltaR_g.append(deltaR[None,:])
                    for r_id in range(60):
                        R_g=self.Rgroup[r_id]
                        anc_local_patch_g.append(transform_points(anc,R_g)[None,:,:])
                        pos_local_patch_g.append(transform_points(pos,R_g)[None,:,:])

                anc_local_patch_g=np.concatenate(anc_local_patch_g,axis=0).astype(np.float32)
                pos_local_patch_g=np.concatenate(pos_local_patch_g,axis=0).astype(np.float32) #(B*60)*1024*3
                rotate_g=np.concatenate(rotate_g,axis=0).astype(np.float32)
                index_g=np.array(index_g)[:,None].astype(np.int)
                deltaR_g=np.concatenate(deltaR_g,axis=0).astype(np.float32)

                anc_local_patch = torch.from_numpy(anc_local_patch_g).cuda()
                pos_local_patch = torch.from_numpy(pos_local_patch_g).cuda()

                # forward
                a_des = self.model(anc_local_patch)
                p_des = self.model(pos_local_patch)
                # descriptor loss
                anc_des = F.normalize(a_des.view(B*60, -1), p=2, dim=1).reshape(B,60,32).permute(0,2,1).cpu() # B*32*60
                pos_des = F.normalize(p_des.view(B*60, -1), p=2, dim=1).reshape(B,60,32).permute(0,2,1).cpu() # B*32*60

                item={
                    'feats0':anc_des, #before enhanced rot
                    'feats1':pos_des, #after enhanced rot
                    'R':torch.from_numpy(rotate_g),
                    'true_idx':torch.from_numpy(index_g),
                    'deltaR':torch.from_numpy(deltaR_g)
                }

                torch.save(item,f'{self.train_batch_dir}/{times*self.yoho_train_num+iter}.pth',_use_new_zipfile_serialization=False)

                del anc_local_patch
                del pos_local_patch


    def yoho_valset_create(self,val_batch_num=3000):
        with torch.no_grad():
            for iter, (anc_local_patch, pos_local_patch, rotate, shift) in enumerate(tqdm(self.val_loader)):
                if iter>val_batch_num:
                    break
                B = anc_local_patch.shape[0]
                anc_local_patch = anc_local_patch.float().numpy()
                pos_local_patch = pos_local_patch.float().numpy()
                rotate = rotate.float().numpy()
                shift = shift.float().numpy()

                anc_local_patch_g=[]
                pos_local_patch_g=[]
                rotate_g=[]
                index_g=[]
                deltaR_g=[]
                for i in range(B):
                    anc=anc_local_patch[i]#1024*3
                    pos=pos_local_patch[i]#1024*3
                    #alignment
                    R_former=rotate[i]#3*3
                    pos=transform_points(pos,R_former)
                    #random rotate
                    R=random_rotation_matrix(180)
                    ind=self.R2DR_id(R)
                    deltaR=self.DeltaR(R,ind)
                    pos=transform_points(pos,R)#对pos random
                    #batch_append
                    rotate_g.append(R[None,:,:])
                    index_g.append(ind)
                    deltaR_g.append(deltaR[None,:])
                    for r_id in range(60):
                        R_g=self.Rgroup[r_id]
                        anc_local_patch_g.append(transform_points(anc,R_g)[None,:,:])
                        pos_local_patch_g.append(transform_points(pos,R_g)[None,:,:])

                anc_local_patch_g=np.concatenate(anc_local_patch_g,axis=0).astype(np.float32)
                pos_local_patch_g=np.concatenate(pos_local_patch_g,axis=0).astype(np.float32) #(B*60)*1024*3
                rotate_g=np.concatenate(rotate_g,axis=0).astype(np.float32)
                index_g=np.array(index_g)[:,None].astype(np.int)
                deltaR_g=np.concatenate(deltaR_g,axis=0).astype(np.float32)

                anc_local_patch = torch.from_numpy(anc_local_patch_g).cuda()
                pos_local_patch = torch.from_numpy(pos_local_patch_g).cuda()

                # forward
                a_des = self.model(anc_local_patch)
                p_des = self.model(pos_local_patch)

                # descriptor loss
                anc_des = F.normalize(a_des.view(B*60, -1), p=2, dim=1).reshape(B,60,32).permute(0,2,1).cpu() # B*32*60
                pos_des = F.normalize(p_des.view(B*60, -1), p=2, dim=1).reshape(B,60,32).permute(0,2,1).cpu() # B*32*60

                item={
                    'feats0':anc_des, #before enhanced rot
                    'feats1':pos_des, #after enhanced rot
                    'R':torch.from_numpy(rotate_g),
                    'true_idx':torch.from_numpy(index_g),
                    'deltaR':torch.from_numpy(deltaR_g)
                }

                torch.save(item,f'{self.val_batch_dir}/{iter}.pth',_use_new_zipfile_serialization=False)

                del anc_local_patch
                del pos_local_patch



if __name__ == '__main__':
    save_train_batch_dir='../data/YOHO_PN/Trainset'
    make_non_exists_dir(save_train_batch_dir)
    save_val_batch_dir='../data/YOHO_PN/valset'
    make_non_exists_dir(save_val_batch_dir)
    batch_size=32  

    trainset_repeat=3
    val_batch_num=100

    creater=Set_creater(save_train_batch_dir,save_val_batch_dir,batch_size)
    for t in range(trainset_repeat):
        creater.yoho_trainset_create(t)
    creater.yoho_valset_create(val_batch_num)
    trainlist=range(creater.yoho_train_num*trainset_repeat)
    vallist=range(val_batch_num)
    save_pickle(trainlist,f'../data/YOHO_PN/train.pkl')
    save_pickle(vallist,f'../data/YOHO_PN/val.pkl')