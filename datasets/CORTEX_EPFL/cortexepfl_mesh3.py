import numpy as np
from skimage import io
from datasets.datasetandsupport import DatasetAndSupport, sample_to_sample_plus

from metrics.standard_metrics import jaccard_index, chamfer_weighted_symmetric
from utils.utils_common import invfreq_lossweights, crop, DataModes, crop_indices, volume_suffix
from utils.utils_mesh import clean_border_pixels, voxel2mesh, sample_outer_surface_in_voxel, normalize_vertices
from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr
# from IPython import embed
import time
from scipy import ndimage

class Sample:
    def __init__(self, x, y, orientation, w, center):
        self.x = x
        self.y = y
        self.w = w
        self.orientation = orientation
        self.center = center

class CortexVoxelDataset(Dataset):

    def __init__(self, data, cfg, mode, base_sparse_plane=None, point_count=None):
        self.data = data
        self.cfg = cfg
        self.mode = mode
        self.base_sparse_plane = base_sparse_plane
        self.point_count = point_count


    def __len__(self):
        return len(self.data)

    def getitem_center(self, idx):
        item = self.data[idx]
        return item.center

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.from_numpy(item.x).cuda()[None]
        y = torch.from_numpy(item.y).cuda().long()

        # embed()
        y[y == 2] = 0
        y[y == 3] = 0
        # y[y==3] = 1

        if self.base_sparse_plane is not None:
            base_plane = torch.from_numpy(self.base_sparse_plane).cuda().float()
        elif self.point_count is not None:
            y_mid = sample_outer_surface_in_voxel(y) ## around 1-2% points 1 among all
            # breakpoint()
            # print(self.point_count)
            idxs = torch.nonzero(y_mid) ## [number of onits x 3] ## all indexes
            y_mid = torch.zeros_like(y)
            perm = torch.randperm(len(idxs)) # random permutation
            idxs = idxs[perm[:self.point_count]]
            y_mid[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
            base_plane = y_mid.clone()
            y = y_mid.clone()
        else:
            base_plane = torch.ones_like(y).float()
            # breakpoint()
        C, D, H, W = x.shape
        center = (D//2, H//2, W//2) 

        # breakpoint()
        y = y.long()

        # embed()

        # from utils.rasterize.rasterize import Rasterize
        # from skimage import io
        # shape = torch.tensor(y.shape)[None].float()
        # gap = 1
        # y_ = clean_border_pixels((y==1).long(), gap=gap)
        # vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)

        # D, H, W = y.shape
        # shape = torch.tensor([D,H,W]).int().cuda()
        # rasterizer = Rasterize(shape)
        # pred_voxels_rasterized = rasterizer(vertices_mc[None], faces_mc[None]).long()
        # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/yy_rast.tif', 255 * np.uint8(pred_voxels_rasterized[0].cpu().data.numpy()))
        # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/yy.tif', 255 * np.uint8(y.cpu().data.numpy()))


        if self.mode == DataModes.TRAINING_EXTENDED: # if training do augmentation

            orientation = torch.tensor([0, -1, 0]).float()
            new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi
            # new_orientation[2] = new_orientation[2] * 0 # no rotation outside x-y plane
            new_orientation = F.normalize(new_orientation, dim=0)
            q = orientation + new_orientation
            q = F.normalize(q, dim=0)
            theta_rotate = stns.stn_quaternion_rotations(q)

            shift = torch.tensor([d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * self.cfg.augmentation_shift_range, y.shape)])
            theta_shift = stns.shift(shift)
            
            f = 0.1
            scale = 1.0 - 2 * f *(torch.rand(1) - 0.5)
            theta_scale = stns.scale(scale) 

            theta = theta_rotate @ theta_shift @ theta_scale
 
            x, y, base_plane = stns.transform(theta, x, y, w=base_plane)
        else:
            pose = torch.zeros(6).cuda()
            # w = torch.zeros_like(y)
            # base_plane = torch.ones_like(y)
            theta = torch.eye(4).cuda()

        # theta = theta.inverse()

        # N, _ = vertices_mc.shape
        # v = torch.cat([vertices_mc, torch.ones(N,1)],dim=1)
        # v = theta[:3] @ v.transpose(1,0)
        # v = v.transpose(1, 0)


        # D, H, W = y.shape
        # shape = torch.tensor([D,H,W]).int().cuda()
        # rasterizer = Rasterize(shape)
        # pred_voxels_rasterized = rasterizer(v[None], faces_mc[None]).long()
        # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/yy_rast_rot.tif', 255 * np.uint8(pred_voxels_rasterized[0].cpu().data.numpy()))
        # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/yy_rot.tif', 255 * np.uint8(y.cpu().data.numpy()))



        x_super_res = torch.tensor(1)
        y_super_res = torch.tensor(1)

        x = crop(x, (C,) + self.cfg.patch_shape, (0,) + center)
        y = crop(y, self.cfg.patch_shape, center) 
        base_plane = crop(base_plane, self.cfg.patch_shape, center)

        surface_points_normalized_all = []
        vertices_mc_all = []
        faces_mc_all = []  

        for i in range(1, self.cfg.num_classes):   
            shape = torch.tensor(y.shape)[None].float()
            if self.mode != DataModes.TRAINING_EXTENDED:
                gap = 1
                y_ = clean_border_pixels((y==i).long(), gap=gap)
                vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
                vertices_mc_all += [vertices_mc]
                faces_mc_all += [faces_mc] 

          
            sphere_vertices = self.cfg.sphere_vertices
            atlas_faces = self.cfg.sphere_faces 
            # self.sphere_vertices = sphere_ssvertices.repeat(self.config.config.batch_size,1,1).float()
          
            p = torch.acos(sphere_vertices[:,2]).cuda()
            t = torch.atan2(sphere_vertices[:,1], sphere_vertices[:,0]).cuda()
            p = torch.tensor(p, requires_grad=True)
            t = torch.tensor(t, requires_grad=True) 
         
            y_outer = sample_outer_surface_in_voxel((y==i).long()) 
            surface_points = torch.nonzero(y_outer)
            surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
            surface_points_normalized = normalize_vertices(surface_points, shape) 
            # surface_points_normalized = y_outer 
           
            # perm = torch.randperm(len(surface_points_normalized))
            N = len(surface_points_normalized)
            point_count = 3000
            idxs = np.arange(N)

            if N >= point_count:
                perm = np.random.choice(idxs,point_count, replace=False)
            else:
                repeats = point_count//N
                vals = []
                for _ in range(repeats):
                    vals += [idxs]

                remainder = point_count - repeats * N
                vals += [np.random.choice(idxs,remainder, replace=False)]
                perm = np.concatenate(vals, axis=0)
            
            surface_points_normalized_all += [surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()]  # randomly pick 3000 points
        
        # print('in')
        if self.mode == DataModes.TRAINING_EXTENDED:
            return {   'x': x, 
                       'faces_atlas': atlas_faces, 
                       'y_voxels': y, 
                       'surface_points': surface_points_normalized_all,
                       'p':p,
                       't':t,
                       'unpool':self.cfg.unpool_indices,
                       'w': y_outer,
                       'theta': theta.inverse()[:3],
                       'base_plane' : base_plane
                    }
        else:
            return {   'x': x,
                       'x_super_res': x_super_res, 
                       'faces_atlas': atlas_faces, 
                       'y_voxels': y,
                       'y_voxels_super_res': y_super_res,
                       'vertices_mc': vertices_mc_all,
                       'faces_mc': faces_mc_all,
                       'surface_points': surface_points_normalized_all,
                       'p':p,
                       't':t,
                       'unpool':[0, 1, 0, 1, 1],
                       'theta': theta.inverse()[:3],
                       'base_plane': base_plane
                    }


class CortexEpfl(DatasetAndSupport):
    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (96, 96, 96), 'Not supported'
        data_root = '/cvlabsrc1'+volume_suffix+'/cvlab/datasets_udaranga/datasets/3d/graham/'
        class_id = 14
        data_version = 'labels_v' + str(class_id) + '/'

        with open(data_root + data_version + 'labels/pre_computed_voxel.pickle', 'rb') as handle:
            data = pickle.load(handle)

        if cfg.sparse_model == 'point_model':
            point_count = 100  ## ex: out of 21k
            print(f"using {point_count} points for annotation")

            data[DataModes.TRAINING_EXTENDED] = CortexVoxelDataset(data[DataModes.TRAINING].data, cfg,\
                                                                   DataModes.TRAINING_EXTENDED, point_count=point_count)
        elif cfg.sparse_model == 'base_plane_model':
            print("Using base plane model for sparse annotation")
            planes_per_axis = 2
            half_plane = len(data[DataModes.TRAINING].data[0].y)//planes_per_axis # 135//2 = 67
            keep_planes = np.zeros_like(data[DataModes.TRAINING].data[0].y)
            keep_planes[:,:,half_plane] = 1
            keep_planes[half_plane,:,:] = 1
            keep_planes[:,half_plane, :] = 1
            # half_plane3 = half_plane*2
            # keep_planes[:,:,half_plane3] = 1
            # keep_planes[half_plane3,:,:] = 1
            # keep_planes[:,half_plane3, :] = 1

            for sample in data[DataModes.TRAINING].data:
                sample.y = sample.y*keep_planes ## for each new training data, only keep few given plane

            data[DataModes.TRAINING_EXTENDED] = CortexVoxelDataset(data[DataModes.TRAINING].data, cfg, \
                                                                   DataModes.TRAINING_EXTENDED, base_sparse_plane=keep_planes)

        else:
            print("with full annotation")
            data[DataModes.TRAINING_EXTENDED] = CortexVoxelDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)

        # data[DataModes.TRAINING_EXTENDED] = CortexVoxelDataset(data[DataModes.TRAINING].data, data[DataModes.TRAINING].cfg, DataModes.TRAINING_EXTENDED)
        # data[DataModes.TRAINING] = CortexVoxelDataset(data[DataModes.TRAINING].data, data[DataModes.TRAINING].cfg, DataModes.TRAINING)
        # data[DataModes.TRAINING_EXTENDED] = CortexVoxelDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)
        # data[DataModes.TRAINING] = CortexVoxelDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING)
        data[DataModes.TESTING] = CortexVoxelDataset(data[DataModes.TESTING].data, cfg, DataModes.TESTING)
    
        return data

    def load_data(self, cfg, trial_id):
        '''
        # Change this to load your training data.

        # pre-synaptic neuron   :   1
        # synapse               :   2
        # post-synaptic neuron  :   3
        # background            :   0
        ''' 

        data_root = '/cvlabsrc1'+volume_suffix+'/cvlab/datasets_udaranga/datasets/3d/graham/'
        class_id = 14
        num_classes = 4
        data_version = 'labels_v' + str(class_id) + '/'
        path_images = data_root + 'imagestack_downscaled.tif'
        path_synapse = data_root + data_version + 'labels/labels_synapses_' + str(class_id) + '.tif'
        path_pre_post = data_root + data_version + 'labels/labels_pre_post_' + str(class_id) + '.tif'


        seeds = data_root + data_version + 'labels/seeds.tif'
        

        ''' Label information '''
        # path_idx = data_root + data_version + 'labels/info.txt'
        # idx = np.loadtxt(path_idx)

        path_idx = data_root + data_version + 'labels/info.npy'
        idx = np.load(path_idx)
         

        ''' Load data '''
        # x = io.imread(path_images)[:200]
        # y_synapses = io.imread(path_synapse)
        # y_pre_post = io.imread(path_pre_post)

        # x = np.float32(x) / 255
        # y_synapses = np.int64(y_synapses)

        # Syn at bottom
        # temp = np.int64(y_pre_post)
        # y_pre_post = np.copy(y_synapses)
        # y_pre_post[temp > 0] = temp[temp > 0]



        x = io.imread(path_images)   
        y_synapses = io.imread(seeds)   

        x = np.float32(x) / 255
        y_synapses = np.int64(y_synapses)
        y_pre_post = y_synapses

        # method 1: split neurons 
        # counts = [[0, 4, 6, 8, 10, 11, 12, 16, 17, 19, 20, 21, 22, 24], range(24, 36), range(24, 36)]

        #  # <<<<<<<<<<<
        counts = [[104, 105, 112, 113, 114, 115, 116, 117, 118, 119, 123, 124,125], 
        [ 1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38, 39, 101, 102, 103, 106, 107, 108, 109, 110, 115, 117, 118, 119, 120, 121, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 140, 141],
        [ 1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38, 39, 101, 102, 103, 106, 107, 108, 109, 110, 115, 117, 118, 119, 120, 121, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 140, 141]
        ]
        data = {}
        patch_shape_extended = tuple([int(np.sqrt(2) * i) + 2 * cfg.augmentation_shift_range for i in cfg.patch_shape]) 
        # volume_shape_extended = tuple([int(np.sqrt(2) * i) + 2 * cfg.augmentation_shift_range for i in x.shape])

        
        # mask = np.zeros_like(x) + 1
        # centre_global = tuple([d//2 for d in x.shape])
        # x_extended = crop(x, volume_shape_extended, centre_global)
        # mask_extended = np.float64(crop(np.uint64(mask), volume_shape_extended, centre_global))
        # mask_extended_dst = ndimage.distance_transform_edt(mask_extended == 0)
        # mask_extended_dst = mask_extended_dst/np.max(mask_extended_dst)
        # x_extended = np.float32(x_extended + mask_extended_dst)

        # x_shape = x.shape
        # del mask_extended
        # del mask_extended_dst
        # del x

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]):

            samples = []
            for j in counts[i]:
                

                # points = np.where(y_synapses == idx[j][1])
                points = np.where(y_synapses == j) # <<<<<<<<<<<
                centre = tuple(np.mean(points, axis=1, dtype=np.int64))


                # extract the object of interesete
                y = np.zeros_like(y_pre_post)
                # for k, id in enumerate(idx[j][:3]):
                #     y[y_pre_post == id] = k + 1
                # y_extended = crop(y, volume_shape_extended, centre_global)
 
 
                 
                # patch_y = crop(y_extended, patch_shape_extended, centre)
                # patch_x = crop(x_extended, patch_shape_extended, centre)

                patch_y = crop(y, patch_shape_extended, centre)
                patch_x = crop(x, patch_shape_extended, centre)
                patch_w = invfreq_lossweights(patch_y, num_classes)

                # Compute orientation

                # First find the Axis
                # syn = patch_y == 2
                # coords = np.array(np.where(syn)).transpose()
                # coords = np.flip(coords, axis=1)  # make it x,y,z
                # syn_center = np.mean(coords, axis=0)
                # pca = PCA(n_components=3)
                # pca.fit(coords)
                # # u = -np.flip(pca.components_)[0]
                # u = pca.components_[2]

                # # Now decide it directed towards pre syn region
                # pre = patch_y == 1
                # coords = np.array(np.where(pre)).transpose()
                # coords = np.flip(coords, axis=1)  # make it x,y,z
                # # pre_center = np.flip(np.mean(coords, axis=0))
                # pre_center = np.mean(coords, axis=0)

                # w = pre_center - syn_center
                # angle = np.arccos(np.dot(u, w) / norm(u) / norm(w)) * 180 / np.pi
                # if angle > 90:
                #     u = -u

                # orientation = u
                orientation = np.array([0,0,0])
 
                np.save(data_root + data_version + datamode + '_' + str(j)  + '_patch_x.npy', patch_x)
                np.save(data_root + data_version + datamode + '_' + str(j)  + '_patch_y.npy', patch_y)
                np.save(data_root + data_version + datamode + '_' + str(j)  + '_patch_w.npy', patch_w)
                np.save(data_root + data_version + datamode + '_' + str(j)  + '_orientation.npy', orientation)
                np.save(data_root + data_version + datamode + '_' + str(j)  + '_centre.npy', centre)

                # samples.append(Sample(patch_x, patch_y, orientation, patch_w, centre))

            # data[datamode] = CortexVoxelDataset(samples, cfg, datamode)

        # data[DataModes.TRAINING_EXTENDED] = data[DataModes.TRAINING]
        # data[DataModes.TRAINING_EXTENDED].mode = DataModes.TRAINING_EXTENDED
        # with open(data_root + data_version + 'labels/pre_computed_voxel_negative_dst_border.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # del y
        # del y_extended

        # print('---')

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]):

            samples = []
            for j in counts[i]:
                # print(j)

                patch_x = np.load(data_root + data_version + datamode + '_' + str(j)  + '_patch_x.npy')
                patch_y = np.load(data_root + data_version + datamode + '_' + str(j)  + '_patch_y.npy')
                patch_w = np.load(data_root + data_version + datamode + '_' + str(j)  + '_patch_w.npy')
                orientation = np.load(data_root + data_version + datamode + '_' + str(j)  + '_orientation.npy')
                centre = np.load(data_root + data_version + datamode + '_' + str(j)  + '_centre.npy')

                samples.append(Sample(patch_x, patch_y, orientation, patch_w, centre))

            data[datamode] = CortexVoxelDataset(samples, cfg, datamode)
        data[DataModes.TRAINING_EXTENDED] = data[DataModes.TRAINING]
        data[DataModes.TRAINING_EXTENDED].mode = DataModes.TRAINING_EXTENDED
        with open(data_root + data_version + 'labels/pre_computed_voxel_456.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(error)
        return data

    def evaluate(self, target, pred, cfg):
        results = {}


        if target.voxel is not None: 
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.mesh is not None:
            target_points = target.points # * cfg.num_classes
            pred_points = pred.mesh #* cfg.num_classes
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        new_value = new_value[DataModes.TESTING][key]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.TESTING][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False




