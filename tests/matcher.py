import os,sys
sys.path.append('..')
import numpy as np
import torch
import tqdm
from utils.utils import make_non_exists_dir
from knn_search import knn_module

class matcher_dual():
    def __init__(self,cfg):
        self.cfg=cfg
        self.KNN=knn_module.KNN(1)

    def match(self,dataset):
        print(f'match the keypoints on {dataset.name}')
        Save_dir=f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match'
        make_non_exists_dir(Save_dir)

        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/Testset/{datasetname}/YOMO_Output_Group_feature'
        for pair in tqdm.tqdm(dataset.pair_ids):
            id0,id1=pair
            #if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            feats0=np.mean(feats0,axis=-1).astype(np.float32)
            feats1=np.mean(feats1,axis=-1).astype(np.float32)
            feats0=torch.from_numpy(np.transpose(feats0)[None,:,:]).cuda()
            feats1=torch.from_numpy(np.transpose(feats1)[None,:,:]).cuda()
            d,argmin_of_0_in_1=self.KNN(feats1,feats0)
            argmin_of_0_in_1=argmin_of_0_in_1[0,0].cpu().numpy()
            d,argmin_of_1_in_0=self.KNN(feats0,feats1)
            argmin_of_1_in_0=argmin_of_1_in_0[0,0].cpu().numpy()
            match_pps=[]
            for i in range(argmin_of_0_in_1.shape[0]):
                in0=i
                in1=argmin_of_0_in_1[i]
                inv_in0=argmin_of_1_in_0[in1]
                if in0==inv_in0:
                    match_pps.append(np.array([[in0,in1]]))
            match_pps=np.concatenate(match_pps,axis=0)
            np.save(f'{Save_dir}/{id0}-{id1}.npy',match_pps)


class matcher_dual_pca():
    def __init__(self,cfg):
        self.cfg=cfg
        self.KNN=knn_module.KNN(1)

    def match(self,dataset):
        print(f'match the keypoints on {dataset.name}')
        Save_dir=f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match'
        make_non_exists_dir(Save_dir)

        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/Testset/{datasetname}/YOMO_Output_Group_feature'
        pca_dir=f'{self.cfg.output_cache_fn}/Testset/{datasetname}/PN_Input_Group_feature/pca'
        for pair in tqdm.tqdm(dataset.pair_ids):
            id0,id1=pair
            if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            pca0=np.load(f'{pca_dir}/{id0}.npy')
            pca1=np.load(f'{pca_dir}/{id1}.npy')
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            ok_mask0=np.where(pca0[:,0]>self.cfg.pca_threshold)[0]
            ok_mask1=np.where(pca1[:,0]>self.cfg.pca_threshold)[0]
            feats0=feats0[ok_mask0]
            feats1=feats1[ok_mask1]
            feats0=np.mean(feats0,axis=-1)
            feats1=np.mean(feats1,axis=-1)
            feats0=torch.from_numpy(np.transpose(feats0.astype(np.float32))[None,:,:]).cuda()
            feats1=torch.from_numpy(np.transpose(feats1.astype(np.float32))[None,:,:]).cuda()
            d,argmin_of_0_in_1=self.KNN(feats1,feats0)
            argmin_of_0_in_1=argmin_of_0_in_1[0,0].cpu().numpy()
            d,argmin_of_1_in_0=self.KNN(feats0,feats1)
            argmin_of_1_in_0=argmin_of_1_in_0[0,0].cpu().numpy()
            match_pps=[]
            for i in range(argmin_of_0_in_1.shape[0]):
                in0=i
                in1=argmin_of_0_in_1[i]
                inv_in0=argmin_of_1_in_0[in1]
                if in0==inv_in0:
                    match_pps.append(np.array([[in0,in1]]))
            match_pps=np.concatenate(match_pps,axis=0)
            match_pps_in_5000=[]
            match_pps_in_5000.append(ok_mask0[match_pps[:,0]][:,None])
            match_pps_in_5000.append(ok_mask1[match_pps[:,1]][:,None])
            match_pps_in_5000=np.concatenate(match_pps_in_5000,axis=1)
            np.save(f'{Save_dir}/{id0}-{id1}.npy',match_pps_in_5000)

name2matcher={
    'Match':matcher_dual,
    'Match_pca':matcher_dual_pca
}