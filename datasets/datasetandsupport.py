from utils.utils_common import crop, DataModes, blend_cpu
import torch
import torch.nn.functional as F
from utils import stns
from utils.utils_mesh import sample_outer_surface, sample_outer_surface_in_voxel, get_extremity_landmarks, voxel2mesh, clean_border_pixels, normalize_vertices
import numpy as np
import os
import pickle
import nibabel as nib
from skimage import io
from tqdm import tqdm
from scipy import ndimage
# from IPython import embed
from skimage import io
from utils import affine_3d_grid_generator
import random  
import h5py

class Sample:
    def __init__(self, x, y, atlas):
        self.x = x
        self.y = y
        self.atlas = atlas

class SamplePlus:
    def __init__(self, x, y, y_outer=None, w=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.w = w
        self.shape = shape


class DatasetAndSupport(object):

    def quick_load_data(self, patch_shape): raise NotImplementedError

    def load_data(self, patch_shape):raise NotImplementedError

    def evaluate(self, target, pred, cfg):raise NotImplementedError

    def save_results(self, target, pred, cfg): raise NotImplementedError

    def update_checkpoint(self, best_so_far, new_value):raise NotImplementedError

def get_grid(D, H, W):
    W, H, D = int(W), int(H), int(D)
    base_grid = torch.zeros((1, D, H, W, 3))
    w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points
    grid = base_grid.cuda()
    return grid

def get_item__(item, mode, config):

    x = item.x.cuda()[None]
    y = item.y.cuda() 
    # x = y[None].float() # <<<<<<<<<<<<< comment
    y_outer = item.y_outer.cuda()
    w = item.w.cuda()
    x_super_res = item.x_super_res[None]
    y_super_res = item.y_super_res 
    # x_super_res = y_super_res[None].float() # <<<<<<<<<<<<< comment
    shape = item.shape 
  
  

    # augmentation done only during training
    if mode == DataModes.TRAINING_EXTENDED:  # if training do augmentation
        if torch.rand(1)[0] > 0.5:
            x = x.permute([0, 1, 3, 2])
            y = y.permute([0, 2, 1])
            y_outer = y_outer.permute([0, 2, 1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
            y_outer = torch.flip(y_outer, dims=[0])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
            y_outer = torch.flip(y_outer, dims=[1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[2])
            y_outer = torch.flip(y_outer, dims=[2])

        orientation = torch.tensor([0, -1, 0]).float()
        new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi
        # new_orientation[2] = new_orientation[2] * 0 # no rotation outside x-y plane
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q)

        shift = torch.tensor([d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * config.augmentation_shift_range, y.shape)])
        theta_shift = stns.shift(shift)
        
        f = 0.1
        scale = 1.0 - 2 * f *(torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale) 

        theta = theta_rotate @ theta_shift @ theta_scale

        # x, y = stns.transform(theta, x, y) 
        x, y, y_outer = stns.transform(theta, x, y, y_outer) 

        # not necessary during training
        x_super_res = None
        y_super_res = None

    # y_outer = sample_outer_surface_in_voxel(y) 
    if mode != DataModes.TRAINING_EXTENDED:
        gap = 1
        y_ = clean_border_pixels(y, gap=gap)
        vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
  
    sphere_vertices = config.sphere_vertices
    atlas_faces = config.sphere_faces 
    # self.sphere_vertices = sphere_vertices.repeat(self.config.config.batch_size,1,1).float()
  
    p = torch.acos(sphere_vertices[:,2]).cuda()
    t = torch.atan2(sphere_vertices[:,1], sphere_vertices[:,0]).cuda()
    p = torch.tensor(p, requires_grad=True)
    t = torch.tensor(t, requires_grad=True) 
 

    surface_points = torch.nonzero(y_outer)
    surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
    surface_points_normalized = normalize_vertices(surface_points, shape) 
    # surface_points_normalized = y_outer 
 
 
    perm = torch.randperm(len(surface_points_normalized))
    point_count = 3000
    surface_points_normalized = surface_points_normalized[perm[:np.min([len(perm), point_count])]]  # randomly pick 3000 points

    if mode == DataModes.TRAINING_EXTENDED:
        return {   'x': x, 
                   'faces_atlas': atlas_faces, 
                   'y_voxels': y, 
                   'surface_points': surface_points_normalized,
                   'p':p,
                   't':t,
                   'unpool':config.unpool_indices,
                   'w': y_outer
                }
    else:
        return {   'x': x,
                   'x_super_res': x_super_res, 
                   'faces_atlas': atlas_faces, 
                   'y_voxels': y,
                   'y_voxels_super_res': y_super_res,
                   'vertices_mc': vertices_mc,
                   'faces_mc': faces_mc,
                   'surface_points': surface_points_normalized,
                   'p':p,
                   't':t,
                   'unpool':[0, 1, 0, 1, 0]}

def get_item(item, mode, config):
 
    x = item.x.cuda()[None]
    y = item.y.cuda() 
    # x = y[None].float() # <<<<<<<<<<<<< comment
    y_outer = item.y_outer.cuda()
    w = item.w.cuda()
    x_super_res = item.x_super_res[None]
    y_super_res = item.y_super_res 
    # x_super_res = y_super_res[None].float() # <<<<<<<<<<<<< comment
    shape = item.shape 
 
  

    # augmentation done only during training
    if mode == DataModes.TRAINING_EXTENDED:  # if training do augmentation
        if torch.rand(1)[0] > 0.5:
            x = x.permute([0, 1, 3, 2])
            y = y.permute([0, 2, 1])
            y_outer = y_outer.permute([0, 2, 1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
            y_outer = torch.flip(y_outer, dims=[0])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
            y_outer = torch.flip(y_outer, dims=[1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[2])
            y_outer = torch.flip(y_outer, dims=[2])

        orientation = torch.tensor([0, -1, 0]).float()
        new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi
        # new_orientation[2] = new_orientation[2] * 0 # no rotation outside x-y plane
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q)

        shift = torch.tensor([d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * config.augmentation_shift_range, y.shape)])
        theta_shift = stns.shift(shift)
        
        f = 0.1
        scale = 1.0 - 2 * f *(torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale) 

        theta = theta_rotate @ theta_shift @ theta_scale

        # x, y = stns.transform(theta, x, y) 
        x, y, y_outer = stns.transform(theta, x, y, y_outer) 

        # not necessary during training

    x_super_res = torch.tensor(1)
    y_super_res = torch.tensor(1)

    surface_points_normalized_all = []
    vertices_mc_all = []
    faces_mc_all = [] 
    for i in range(1, config.num_classes):   
        shape = torch.tensor(y.shape)[None].float()
        if mode != DataModes.TRAINING_EXTENDED:
            gap = 1
            y_ = clean_border_pixels((y==i).long(), gap=gap)
            vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
            vertices_mc_all += [vertices_mc]
            faces_mc_all += [faces_mc]
      
        sphere_vertices = config.sphere_vertices
        atlas_faces = config.sphere_faces 
        # self.sphere_vertices = sphere_vertices.repeat(self.config.config.batch_size,1,1).float()
      
        p = torch.acos(sphere_vertices[:,2]).cuda()
        t = torch.atan2(sphere_vertices[:,1], sphere_vertices[:,0]).cuda()
        p = torch.tensor(p, requires_grad=True)
        t = torch.tensor(t, requires_grad=True) 
     
        y_outer = sample_outer_surface_in_voxel((y==i).long()) 
        surface_points = torch.nonzero(y_outer)
        surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
        surface_points_normalized = normalize_vertices(surface_points, shape) 
        # surface_points_normalized = y_outer 
      
        
        perm = torch.randperm(len(surface_points_normalized))
        point_count = 2000
        surface_points_normalized_all += [surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()]  # randomly pick 3000 points
     
    if mode == DataModes.TRAINING_EXTENDED:
        return {   'x': x, 
                   'faces_atlas': atlas_faces, 
                   'y_voxels': y, 
                   'surface_points': surface_points_normalized_all,
                   'p':p,
                   't':t,
                   'unpool':config.unpool_indices,
                   'w': y_outer
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
                   'unpool':[0, 1, 1, 1, 0]}

def get_item_(item, mode, config):
    x = item.x.cuda()[None]
    y = item.y.cuda() 
    # x = y[None] # <<<<<<<<<<<<< comment
    y_outer = item.y_outer.cuda()
    w = item.w.cuda()
    x_super_res = item.x_super_res[None]
    y_super_res = item.y_super_res 
    shape = item.shape

    # print('in')
    # x_temp = x.clone()
    # y_temp = y.clone()

    # embed()
    # x = x_temp
    # y = y_temp
    surface_points = y_outer
    # surface_points_before = torch.nonzero(y_outer)
    # surface_points_before = torch.flip(surface_points_before, dims=[1])
    # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/check300.tif', np.uint8(y_outer.data.cpu().numpy() * 255))
     
    # surface_points = y_outer
    # augmentation done only during training
    if mode == DataModes.TRAINING_EXTENDED:  # if training do augmentation 
        if torch.rand(1)[0] > 0.0:
            x = x.permute([0, 1, 3, 2])
            y = y.permute([0, 2, 1]) 
            surface_points = surface_points[:,[1,0,2]]

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0]) 
            surface_points[:,2] = -surface_points[:,2]

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1]) 
            surface_points[:,1] = -surface_points[:,1]

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[2]) 
            surface_points[:,0] = -surface_points[:,0]

        orientation = torch.tensor([0, -1, 0]).float()
        new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi 
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q)

        shift = torch.tensor([d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * config.augmentation_shift_range, y.shape)])
        D,H,W = y.shape
        # shift = torch.tensor([10,15,20]).float() / D
        theta_shift = stns.shift(shift)

        f = 0.1
        scale = 1.0 - 2 * f *(torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale) 

        theta = theta_rotate @ theta_shift @ theta_scale

        x, y, w = stns.transform(theta, x, y, w)  

 

        # not necessary during training
        x_super_res = None
        y_super_res = None
  
        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/check307.tif', np.uint8(y_outer_grid_sampler.data.cpu().numpy() * 255))
    
        # theta_shift = stns.shift(torch.tensor([10,0,0]))

        # surface_points_after = surface_points_before.float() - (shape.cuda()-1)/2
        
        theta_inv =  theta_scale.inverse() @ theta_shift.inverse() @ theta_rotate.inverse()
        # theta_inv = theta_rotate
        theta_inv = theta_inv[:3] 
        surface_points = torch.cat([surface_points, torch.ones(len(surface_points),1).cuda()],dim=1)
        surface_points = theta_inv.cuda() @ surface_points.float().permute(1, 0)  
        surface_points = surface_points.permute(1, 0)
        # surface_points_after = surface_points_after.float() @ theta_rotate_1.cuda() 
        # surface_points_after = surface_points_after + (shape.cuda()-1)/2
        # surface_points = torch.round(surface_points_after).long()
  
    # embed()  
    # print('{} {}'.format(torch.any(surface_points<-1), torch.any(surface_points>1)), end='') 
    surface_points = surface_points[torch.all(surface_points>-1,dim=1)  * torch.all(surface_points<1,dim=1)] 
    # print(' | {} {}'.format(torch.any(surface_points<-1), torch.any(surface_points>1)))

    # vertices_ = torch.round((surface_points + 1)*63.0/2).long()
    # y_outer_ = torch.zeros_like(y)
    # y_outer_[vertices_[:,2], vertices_[:,1], vertices_[:,0]] = 1
    # y_outer_ = y_outer_ + 3*y
    
    # x_ = (x - x.min())/(x.max()-x.min()) 
    # overlay_y_hat = blend_cpu(x_[0].cpu(), y_outer_.cpu(), 4)
    # x_ = x_[0]
    # x_ = 255*x_[:,:,:,None].repeat(1,1,1,3).cpu()
    # overlay = np.concatenate([x_, overlay_y_hat], axis=2) 
    # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/check_{}.tif'.format(int(torch.rand(1)*10000)), np.uint8(overlay))
 
    # print(crash)
    gap = 1 
    y_ = clean_border_pixels(y, gap=gap)
    vertices_mc, faces_mc = voxel2mesh(y_, gap, torch.tensor(y.shape)[None].float())
 
    sphere_vertices = config.sphere_vertices
    sphere_faces = config.sphere_faces 
    # self.sphere_vertices = sphere_vertices.repeat(self.config.config.batch_size,1,1).float()
  
    p = torch.acos(sphere_vertices[:,2]) 
    t = torch.atan2(sphere_vertices[:,1], sphere_vertices[:,0]) 
    p = torch.tensor(p, requires_grad=True)
    t = torch.tensor(t, requires_grad=True) 

    # # points on sphere
    # x_ = torch.sin(p)*torch.cos(t)
    # y_ = torch.sin(p)*torch.sin(t)
    # z_ = torch.cos(p)
    # atlas_vertices = torch.cat([x_[:,None],y_[:,None],z_[:,None]],dim=1).float()
  
    # surface_points = torch.nonzero(y_outer)
     
    # surface_points = normalize_vertices(surface_points, shape)  


 

    if mode == DataModes.TRAINING_EXTENDED:
        return {   'x': x, 
                   'faces_atlas': sphere_faces, 
                   'y_voxels': y, 
                   'surface_points': surface_points,
                   'p':p,
                   't':t,
                   'unpool':config.unpool_indices
                }
    else:
        return {   'x': x,
                   'x_super_res': x_super_res, 
                   'faces_atlas': sphere_faces, 
                   'y_voxels': y,
                   'y_voxels_super_res': y_super_res,
                   'vertices_mc': vertices_mc,
                   'faces_mc': faces_mc,
                   'surface_points': surface_points,
                   'p':p,
                   't':t,
                   'unpool':[1, 1, 1, 0, 0]}
 

def read_sample(data_root, sample, out_shape, pad_shape):
    x = np.load('{}/imagesTr/{}'.format(data_root, sample))
    y = np.load('{}/labelsTr/{}'.format(data_root, sample))

    D, H, W = x.shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    D, H, W = pad_shape
    x = crop(x, (D, H, W), (center_z, center_y, center_x))
    y = crop(y, (D, H, W), (center_z, center_y, center_x))
 
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    x = F.interpolate(x[None, None], out_shape, mode='trilinear', align_corners=False)[0, 0]
    y = F.interpolate(y[None, None].float(), out_shape, mode='nearest')[0, 0].long()

    return x, y

def dataset_init(data_root, multi_stack, CT=False):

    samples = [dir for dir in os.listdir('{}/imagesTr'.format(data_root))]

    inputs = []
    labels = []
    real_sizes = []
    file_names = []

    vals = []
    sizes = []
    for itr, sample in tqdm(enumerate(samples)):
        if '.nii.gz' in sample and '._' not in sample and '.npy' not in sample and '.tif' not in sample:
            x = nib.load('{}/imagesTr/{}'.format(data_root, sample))
            y = nib.load('{}/labelsTr/{}'.format(data_root, sample)).get_fdata() > 0
            resolution = np.diag(x.header.get_sform())[:3]
            x = x.get_fdata()
            if multi_stack is not None:
                x = x[:, :, :, multi_stack]
            real_size = np.round(np.array(x.shape) * resolution)

            # inputs.append(x)
            # labels.append(y)
            # real_sizes.append(real_size)
            # file_names.append(sample)
            file_name = sample
 

            x = torch.from_numpy(x).permute([2, 1, 0]).cuda().float()
            y = torch.from_numpy(y).permute([2, 1, 0]).cuda().float()
            #
            W, H, D = real_size
            W, H, D = int(W), int(H), int(D)
            base_grid = torch.zeros((1, D, H, W, 3))
            w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
            d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 0] = w_points
            base_grid[:, :, :, :, 1] = h_points
            base_grid[:, :, :, :, 2] = d_points
            grid = base_grid.cuda()

            # sizes += [real_size]
            # print('{},{},{};'.format(real_size[0],real_size[1],real_size[2]))

            x = F.grid_sample(x[None, None], grid, mode='bilinear', padding_mode='border')[0, 0].cpu().numpy()
            y = F.grid_sample(y[None, None], grid, mode='nearest', padding_mode='border')[0, 0].long().cpu().numpy()


            x = (x - np.mean(x))/np.std(x)

            np.save('{}/imagesTr/{}'.format(data_root, file_name), x)
            np.save('{}/labelsTr/{}'.format(data_root, file_name), y)

            # x = (x - np.min(x)) / (np.max(x) - np.min(x))
            # io.imsave('{}/imagesTr/{}.tif'.format(data_root, file_name), np.uint8(255 * x))
            # io.imsave('{}/labelsTr/{}.tif'.format(data_root, file_name), np.uint8(255 * y))

        # center_z, center_y, center_x = D // 2, H // 2, W // 2
            # D, H, W = largest_size
            # x = crop(x, (D, H, W), (center_z, center_y, center_x))
            # y = crop(y, (D, H, W), (center_z, center_y, center_x))




def load_nii(data_root, cfg, Dataset, output_shape, pad_shape, multi_stack=None, CT=False):
    # dataset_init(data_root, multi_stack, CT)
    # raise Exception()
    samples = [dir for dir in os.listdir('{}/imagesTr'.format(data_root))]

    inputs = []
    labels = []

    vals = []
    sizes = []
    print('start')
    for itr, sample in enumerate(samples):

        if '.npy' in sample and '._' not in sample:

            x, y = read_sample(data_root, sample, output_shape, pad_shape)
            inputs += [x.cpu()]
            labels += [y.cpu()]

            # if itr == 50:
            #     break
    # sizes = np.array(sizes)
    
    inputs_ = [i[None].data.numpy() for i in inputs] 
    labels_ = [i[None].data.numpy() for i in labels]
    inputs_ = np.concatenate(inputs_, axis=0)
    labels_ = np.concatenate(labels_, axis=0)

    hf = h5py.File(data_root + '/data.h5', 'w') 
    hf.create_dataset('inputs', data=inputs_)
    hf.create_dataset('labels', data=labels_)
    hf.close()


    # hf = h5py.File(data_root + '/data.h5', 'r')
    # inputs = hf.get('inputs') 
    # labels = hf.get('labels')
    # hf.close() 
    raise Exception() 
    print('')
    print(len(inputs))
    print(inputs[0].shape)
    print(len(labels))
    print('')
    
    np.random.seed(0)
    perm = np.random.permutation(len(inputs))
    tr_length = cfg.training_set_size
    counts = [perm[:len(inputs)//2], perm[len(inputs)//2:]]

    atlas = labels[perm[0]]

    data = {}
    # down_sample_shape = (32, 128, 128)
    for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):

        samples = []

        for j in counts[i]:
            x = inputs[j]
            y = labels[j]

            samples.append(Sample(x, y, atlas))

        with open(data_root + '/pre_loaded_data_' + datamode + '.pickle', 'wb') as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data[datamode] = Dataset(samples, cfg, datamode)

    data[DataModes.TRAINING_EXTENDED] = Dataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)
    data[DataModes.VALIDATION] = data[DataModes.TESTING]

    return data

def sample_to_sample_plus(samples, cfg, datamode):

    new_samples = []
    # surface_point_count = 100
    for sample in samples: 
        if cfg.low_resolution is not None:
            x_super_res = sample.x.cuda().float()
            y_super_res = sample.y.cuda().long() 


            high_res, _, _ = x_super_res.shape
            D = high_res//cfg.low_resolution[0]
            K = torch.zeros(1,1,D,D,D).cuda().float()
            K[0,0,D//2-1:D//2+1,D//2-1:D//2+1,D//2-1:D//2+1] = 1 

            x = F.conv3d(x_super_res[None, None], K,bias=None, stride=D )[0, 0] 
            y = (F.conv3d(y_super_res[None, None].float(), K,bias=None, stride=D )[0,0]>4).long()
            # x_       = F.interpolate(x_super_res[None, None], cfg.low_resolution, mode='trilinear')[0, 0]
            # y_       = F.interpolate(y_super_res[None, None].float(), cfg.low_resolution, mode='nearest').long()[0, 0]

            # embed()  

            y_outer = sample_outer_surface_in_voxel(y_super_res)
            shape = torch.tensor(y_super_res.shape)[None].float()
            y_outer = torch.nonzero(y_outer)
            y_outer = torch.flip(y_outer, dims=[1]) # x,y,z
            y_outer = normalize_vertices(y_outer, shape)

            # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/check1006.tif', np.uint8(x_super_res.data.cpu().numpy() * 255)) 
            # vertices_ = torch.floor(y_outer * 32 + 32).long() 
            # # vertices_ = torch.floor(y_outer.float()/6).long()
            # y_outer_ = torch.zeros_like(y)
            # y_outer_[vertices_[:,2], vertices_[:,1], vertices_[:,0]] = 1
            # y_outer_ = y_outer_ + 3*y
            # y_outer_ = y_outer_.data.cpu().numpy()
            # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/check1006.tif', np.uint8(y_outer_ * 63)) 

            high_res, _, _ = x_super_res.shape
            D = high_res//64
            K = torch.zeros(1,1,D,D,D).cuda().float()
            K[0,0,D//2-1:D//2+1,D//2-1:D//2+1,D//2-1:D//2+1] = 1 

            x_super_res = F.conv3d(x_super_res[None, None], K,bias=None, stride=D )[0, 0]
            y_super_res = (F.conv3d(y_super_res[None, None].float(), K,bias=None, stride=D )[0,0]>4).long()
            print(crash)
            # y_super_res = y_super_res.long()
            
        else:
            x = sample.x
            y = sample.y 
 
            y = (y>0).long()


            # y_outer = sample_outer_surface_in_voxel(y) 
            # y_outer = torch.nonzero(y_outer) 

            center = tuple([d // 2 for d in x.shape]) 
            x = crop(x, cfg.hint_patch_shape, center) 
            y = crop(y, cfg.hint_patch_shape, center)   


            
            shape = torch.tensor(y.shape)[None].float()
            y_outer = sample_outer_surface_in_voxel(y)  

            # point_count = 100
            # # print(point_count)
            # idxs = torch.nonzero(border)
            # y_outer = torch.zeros_like(y)
            # perm = torch.randperm(len(idxs)) 
            # idxs = idxs[perm[:point_count]] 
            # y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1

            # if datamode == DataModes.TRAINING: 
            #     D,H,W = y.shape

            #     start = None
            #     end = None
            #     for k in range(D):
            #         if start is None and torch.sum(y[k]) > 0:
            #             start = k

            #         if start is not None and end is None and torch.sum(y[k]) == 0:
            #             end = k

            #     slc = random.randint(start,end)
 
            #     y_outer = torch.zeros_like(y) 
            #     y_outer[slc] = 1

            #     temp = torch.zeros_like(y)
            #     temp[slc] = y[slc]
            #     y = temp 

            # else:
            #     y_outer = y



            # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/y.tif', 255*np.uint8(data['y_voxels'].data.cpu().numpy()))   
            # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/y.tif', 255*np.uint8(y))
            # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/y_outer.tif', 255*np.uint8(y_outer))
            # io.imsave('/cvlabdata1/cvlab/datasets_udaranga/check.tif', 255*np.uint8(y_outer))
  
            # y_outer = sample_outer_surface_in_voxel(y) 
            # y_outer = torch.nonzero(y_outer)
            # y_outer = torch.flip(y_outer, dims=[1]) # x,y,z 
            # y_outer = normalize_vertices(y_outer, shape)

            # perm = torch.randperm(len(y_outer)) 
            # point_count = 500
            # y_outer = y_outer[perm[:np.min([len(perm), point_count])]]  # randomly pick 3000 points

            x_super_res = torch.tensor(1)
            y_super_res = torch.tensor(1)

        # w = torch.zeros_like(y) 
        w = torch.tensor(1)

        # y_dst = ndimage.distance_transform_edt(1-y_outer.data.cpu().numpy()) #/ center[0]
        # y_dst = torch.from_numpy(y_dst).float()[None].cuda()
       
        new_samples += [SamplePlus(x.cpu(), y.cpu(), y_outer=y_outer.cpu(), w=w.cpu(), x_super_res=x_super_res.cpu(), y_super_res=y_super_res.cpu(), shape=shape)]

    return new_samples