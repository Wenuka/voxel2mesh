import numpy as np
from skimage import io
from datasets.datasetandsupport import DatasetAndSupport, get_item, SamplePlus, sample_to_sample_plus

# from evaluate.standard_metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.utils_common import invfreq_lossweights, volume_suffix, crop, DataModes, crop_indices, blend
from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels, sample_outer_surface_in_voxel, normalize_vertices 
 
from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr
import torch
from scipy import ndimage
import os
# from IPython import embed
import pydicom

class Sample:
    def __init__(self, x, y, atlas=None):
        self.x = x
        self.y = y
        self.atlas = atlas
  
class ChaosDataset(Dataset):
    def __init__(self, data, cfg, mode, base_sparse_plane=None, point_model=None):
        self.data = data
        self.cfg = cfg
        self.mode = mode
        self.base_sparse_plane = base_sparse_plane
        self.point_model = point_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        while True:
            x = torch.from_numpy(item.x).cuda()[None]
            y = torch.from_numpy(item.y).cuda().long()
            # y[y == 2] = 0 ## now y==2 means inside points
            y[y == 3] = 0
            # y[y==3] = 1
            if self.base_sparse_plane is not None:
                base_plane = torch.from_numpy(self.base_sparse_plane[idx]).cuda().float()
            else:
                base_plane = torch.ones_like(y).float()
            # breakpoint()
            C, D, H, W = x.shape
            center = (D//2, H//2, W//2)
            y = y.long()

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

            x_super_res = torch.tensor(1)
            y_super_res = torch.tensor(1)

            x = crop(x, (C,) + self.cfg.patch_shape, (0,) + center)
            y = crop(y, self.cfg.patch_shape, center)
            base_plane = crop(base_plane, self.cfg.patch_shape, center)


            ## change for model_id = 4
            if self.point_model is not None:
                surface_points = torch.nonzero((y == 1))
                y_outer = torch.zeros_like(y)
                y_outer[surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]] = 1
                y[y == 2] = 1

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

                ## change for model_id = 4
                if self.point_model is None:

                    y_outer = sample_outer_surface_in_voxel((y==i).long())
                    surface_points = torch.nonzero(y_outer)

                surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
                surface_points_normalized = normalize_vertices(surface_points, shape)
                # surface_points_normalized = y_outer

                # perm = torch.randperm(len(surface_points_normalized))
                N = len(surface_points_normalized)

                surface_points_normalized_all += [surface_points_normalized.cuda()]
            if N > 0:
                break
            else:
                print("re-applying deformation coz N=0")

        # print('in')
        # breakpoint()
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

class Chaos(DatasetAndSupport):

    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def select_min_max_lines(self, y_one_plane, keep_plane, half_width=5):
        # keep_plane = np.zeros_like(y_one_plane.cpu())
        idxs = torch.nonzero((y_one_plane == 1))
        if len(idxs)>0:
            idxs_argsort = torch.argsort(idxs, dim=0)
            xmin, ymin = idxs_argsort[0]
            xmax, ymax = idxs_argsort[-1]
            for val in [xmin, ymin, xmax, ymax]:
                keep_plane[(idxs[val][0] - half_width):(idxs[val][0] + half_width),
                (idxs[val][1] - half_width):(idxs[val][1] + half_width)] = 1
        return keep_plane

    def select_few_lines(self, y_one_plane, keep_plane, line_count, half_width=5, select_type="on_sort"):
        idxs = torch.nonzero((y_one_plane == 1))
        # breakpoint()
        # select_type = "on_sort"  # "random", "on_sort", "on_sort_y"
        if len(idxs) > 0:
            if (select_type=="random"):
                perm = torch.randperm(len(idxs))
                idxs_points = idxs[perm[:line_count]]
            else:
            # elif (select_type=="on_sort"):
                idxs_argsort = torch.argsort(idxs, dim=0)
                idxs_points = []
                # breakpoint()
                for i in range(0, len(idxs), max((len(idxs)*2)//(line_count-1), 1)):
                    idxs_points.append(idxs[idxs_argsort[i][0]])  # add from sorted x
                    idxs_points.append(idxs[idxs_argsort[i][1]])  # add from sorted y

                idxs_points.append(idxs[idxs_argsort[-1][0]])  # add last from sorted x
                idxs_points.append(idxs[idxs_argsort[-1][1]])  # add last from sorted y

            for val in idxs_points:
                keep_plane[(val[0] - half_width):(val[0] + half_width),
                    (val[1] - half_width):(val[1] + half_width)] = 1
            # breakpoint()

        return keep_plane

    def process_for_quick_load(self, data, cfg):

        ### change for model_id = 1
        for sample in data[DataModes.TRAINING].data:
            sample.y[sample.y >1] = 0
        for sample in data[DataModes.TESTING].data:
            sample.y[sample.y >1] = 0
        # breakpoint()
        if cfg.sparse_model == 'line_model':
            half_width = cfg.half_width
            line_count_per_plane = cfg.line_count_per_plane
            select_type = cfg.line_select_type
            # half_width = 4
            # half_height = 5
            print(f"Using line model for sparse annotation with half width: {half_width}")
            # half_plane = len(data[DataModes.TRAINING].data[0].y) // 2  # 135//2 = 67
            lenlist = []
            base_plane_list = []
            for sample in data[DataModes.TRAINING].data:
                keep_lines = np.zeros_like(data[DataModes.TRAINING].data[0].y)
                half_plane_org = len(data[DataModes.TRAINING].data[0].y) // 4

                y_val = torch.from_numpy(sample.y).cuda().long()
                y_outer = sample_outer_surface_in_voxel(y_val)
                # breakpoint()
                for half_plane in [half_plane_org, half_plane_org*2, half_plane_org*3]:
                    if (select_type == "minmax"):
                        keep_lines[:, :, half_plane] = self.select_min_max_lines(y_outer[:, :, half_plane], keep_lines[:, :, half_plane], half_width=half_width)
                        keep_lines[half_plane, :, :] = self.select_min_max_lines(y_outer[half_plane, :, :], keep_lines[half_plane, :, :], half_width=half_width)
                        keep_lines[:, half_plane, :] = self.select_min_max_lines(y_outer[:, half_plane, :], keep_lines[:, half_plane, :], half_width=half_width)

                    else:
                        keep_lines[:, :, half_plane] = self.select_few_lines(y_outer[:, :, half_plane], keep_lines[:, :, half_plane], line_count_per_plane, half_width=half_width, select_type=select_type)
                        keep_lines[half_plane, :, :] = self.select_few_lines(y_outer[half_plane, :, :], keep_lines[half_plane, :, :], line_count_per_plane, half_width=half_width, select_type=select_type)
                        keep_lines[:, half_plane, :] = self.select_few_lines(y_outer[:, half_plane, :], keep_lines[:, half_plane, :], line_count_per_plane, half_width=half_width, select_type=select_type)



                sample.y = y_outer.cpu().float().detach().numpy() * keep_lines
                base_plane_list.append(sample.y)

                lenlist.append(sample.y.sum())
            print(f"Average points of a plane {(sum(lenlist) / len(lenlist))}")
            print(lenlist)
            # io.imsave('y.tif', torch.from_numpy(sample.y).cpu().float().detach().numpy())
            # breakpoint()

            # io.imsave('y_outer.tif', y_outer.cpu().float().detach().numpy())

            data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg, \
                                                                   DataModes.TRAINING_EXTENDED, point_model=True,
                                                                   base_sparse_plane=base_plane_list)

        elif cfg.sparse_model == 'point_model':
            point_count_org = cfg.point_count
            zero_points = cfg.zero_points
            inside_points = cfg.inside_points

            if inside_points:
                point_count_org = int(point_count_org / 2)
                point_count2 = int(point_count_org / 2)
                print(f"using {point_count_org} points for annotation and {point_count_org/2} for inside and background")
            elif zero_points:
                point_count_org = int(point_count_org / 2)
                print(f"using {point_count_org} points for each annotation and background")
            else:
                print(f"using {point_count_org} points for annotation")

            base_plane_list = []
            for sample in data[DataModes.TRAINING].data:
                point_count = point_count_org
                yy = torch.from_numpy(sample.y).cuda().long()
                y_mid = sample_outer_surface_in_voxel(yy)
                idxs = torch.nonzero((y_mid == 1))
                zero_idxs = torch.nonzero((yy == 0))
                ## todo: remove if idx points in zero_idxs
                y_mid = torch.zeros_like(yy)
                perm = torch.randperm(len(idxs))  # random permutation
                zero_perm = torch.randperm(len(zero_idxs)) # random permutation
                idxs = idxs[perm[:point_count]]
                y_mid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
                base_plane = y_mid.clone()
                if inside_points:
                    point_count = point_count2 # now its 25% from original
                    inside_idxs = torch.nonzero((yy == 1))
                    inside_perm = torch.randperm(len(inside_idxs)) # random permutation
                    inside_idxs = inside_idxs[inside_perm[:point_count]]
                    y_mid[inside_idxs[:, 0], inside_idxs[:, 1], inside_idxs[:, 2]] = 2
                    base_plane[inside_idxs[:,0], inside_idxs[:,1], inside_idxs[:,2]] = 1
                if zero_points:
                    zero_idxs = zero_idxs[zero_perm[:point_count]]
                    base_plane[zero_idxs[:,0], zero_idxs[:,1], zero_idxs[:,2]] = 1
                sample.y = y_mid.clone().cpu().numpy()
                base_plane_list.append(base_plane.cpu().numpy())
                # io.imsave('data_org_Y.tif', sample.y.cpu().float().detach().numpy())
            # breakpoint()

            data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg,\
                                                                   DataModes.TRAINING_EXTENDED, point_model=True, base_sparse_plane=base_plane_list)
        elif cfg.sparse_model == 'hybrid_model':
            point_count_org = cfg.point_count
            zero_points = cfg.zero_points
            inside_points = cfg.inside_points

            half_plane = len(data[DataModes.TRAINING].data[0].y) // 2  # 135//2 = 67
            keep_planes = np.zeros_like(data[DataModes.TRAINING].data[0].y)
            keep_planes[:, :, half_plane] = 1
            # keep_planes[half_plane, :, :] = 1
            # keep_planes[:, half_plane, :] = 1


            if inside_points:
                point_count_org = int(point_count_org / 2)
                point_count2 = int(point_count_org / 2)
                print(f"using {point_count_org} points for annotation and {point_count_org/2} for inside and background")
            elif zero_points:
                point_count_org = int(point_count_org / 2)
                print(f"using {point_count_org} points for each annotation and background")
            else:
                print(f"using {point_count_org} points for annotation")

            base_plane_list = []
            for sample in data[DataModes.TRAINING].data:
                point_count = point_count_org
                yy = torch.from_numpy(sample.y).cuda().long()
                y_outer = sample_outer_surface_in_voxel(yy)
                idxs = torch.nonzero((y_outer == 1))
                zero_idxs = torch.nonzero((yy == 0))
                ## todo: remove if idx points in zero_idxs

                y_mid = yy * torch.from_numpy(keep_planes).cuda().long() *2 #inside points are 2
                # y_mid = y_mid *2 #inside points are 2
                y_outer = y_outer * torch.from_numpy(keep_planes).cuda().long()
                line_idxs = torch.nonzero((y_outer == 1))
                y_mid[line_idxs[:, 0], line_idxs[:, 1], line_idxs[:, 2]] = 1 # outer line should be 1

                perm = torch.randperm(len(idxs))  # random permutation
                zero_perm = torch.randperm(len(zero_idxs)) # random permutation
                idxs = idxs[perm[:point_count]]
                y_mid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
                base_plane = torch.from_numpy(keep_planes).cuda().clone()
                base_plane[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1

                if inside_points:
                    point_count = point_count2 # now its 25% from original
                    inside_idxs = torch.nonzero((yy == 1))
                    inside_perm = torch.randperm(len(inside_idxs)) # random permutation
                    inside_idxs = inside_idxs[inside_perm[:point_count]]
                    y_mid[inside_idxs[:, 0], inside_idxs[:, 1], inside_idxs[:, 2]] = 2
                    base_plane[inside_idxs[:,0], inside_idxs[:,1], inside_idxs[:,2]] = 1
                if zero_points:
                    zero_idxs = zero_idxs[zero_perm[:point_count]]
                    base_plane[zero_idxs[:,0], zero_idxs[:,1], zero_idxs[:,2]] = 1
                sample.y = y_mid.clone().cpu().numpy()
                base_plane_list.append(base_plane.cpu().numpy())
                # io.imsave('y_mid.tif', y_mid.cpu().float().detach().numpy())
            breakpoint()

            data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg,\
                                                                   DataModes.TRAINING_EXTENDED, point_model=True, base_sparse_plane=base_plane_list)

        elif cfg.sparse_model == 'base_plane_model':
            print("Using base plane model for sparse annotation")
            half_plane = len(data[DataModes.TRAINING].data[0].y)//2 # 128//2 = 64
            # half_plane3 = len(data[DataModes.TRAINING].data[0].y)//3
            keep_planes = np.zeros_like(data[DataModes.TRAINING].data[0].y)
            keep_planes[:,:,half_plane] = 1
            # keep_planes[half_plane,:,:] = 1
            # keep_planes[:,half_plane, :] = 1
            # half_plane3 = half_plane3*2
            # keep_planes[:,:,half_plane3] = 1
            # keep_planes[half_plane3,:,:] = 1
            # keep_planes[:,half_plane3, :] = 1
            lenlist = []
            base_plane_list = []
            for sample in data[DataModes.TRAINING].data:
                yyy = torch.from_numpy(sample.y).cuda().long()
                y_plane = sample_outer_surface_in_voxel(yyy)
                y_plane = y_plane*torch.from_numpy(keep_planes).cuda().long()
                sample.y = sample.y*keep_planes ## for each new training data, only keep few given plane
                base_plane_list.append(keep_planes)
                # yyy = torch.from_numpy(sample.y).cuda().long()
                # y_plane_sum = sample_outer_surface_in_voxel(yyy).sum()
                # lenlist.append((y_plane_sum - 2*yyy.sum()).item())
                lenlist.append(y_plane.sum().item())
            print(f"Average points of a plane {(sum(lenlist) / len(lenlist))}")
            print(lenlist)
            # breakpoint()

            data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg, \
                                                                   DataModes.TRAINING_EXTENDED, base_sparse_plane=base_plane_list)

        else:
            print("with full annotation")
            data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)

        data[DataModes.TESTING] = ChaosDataset(data[DataModes.TESTING].data, cfg, DataModes.TESTING)

    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.hint_patch_shape
        data_root = '/cvlabsrc1'+volume_suffix+'/cvlab/datasets_udaranga/datasets/3d/chaos/Train_Sets/CT'

        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            # breakpoint()
            with open(data_root + '/pre_loaded_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                samples = pickle.load(handle)
                # if datamode == DataModes.TRAINING:
                #     samples = [samples[0]]
                # if datamode == DataModes.TESTING:
                #     samples = samples[:5]
                # new_samples = sample_to_sample_plus(samples, cfg, datamode)
                for sample in samples:
                    sample.y = sample.y.cpu().float().detach().numpy()
                    sample.x = sample.x.cpu().float().detach().numpy()
                data[datamode] = ChaosDataset(samples, cfg, datamode)
                # io.imsave('y2.tif', torch.from_numpy(data['training'].data[-1].y).cpu().float().detach().numpy())

        self.process_for_quick_load(data, cfg)
        # breakpoint()
        io.imsave(f'y_last_Ex{cfg.experiment_idx}_{trial_id}.tif', torch.from_numpy(data['training'].data[-1].y).cpu().float().detach().numpy())

        # data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)
        # data[DataModes.VALIDATION] = data[DataModes.TESTING] ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< switch to testing
        return data

    def load_data(self, cfg, trial_id):
        '''
         :
        '''

        data_root = '/cvlabsrc1'+volume_suffix+'/cvlab/datasets_udaranga/datasets/3d/chaos/Train_Sets/CT'
        samples = [dir for dir in os.listdir(data_root)]
 
        pad_shape = (384, 384, 384)
        inputs = []
        labels = []

        for sample in samples:
            if 'pickle' not in sample:
                print(sample)
                x = [] 
                images_path = [dir for dir in os.listdir('{}/{}/DICOM_anon'.format(data_root, sample)) if 'dcm' in dir]
                for image_path in images_path:
                    file = pydicom.dcmread('{}/{}/DICOM_anon/{}'.format(data_root, sample, image_path))
                    x += [file.pixel_array] 

                d_resolution = file.SliceThickness
                h_resolution, w_resolution = file.PixelSpacing 
                x = np.float32(np.array(x))

                # clip: x 
                # CHAOS CHALLENGE: MedianCHAOS
                # Vladimir Groza from Median Technologies: CHAOS 1st place solution overview.
                # embed()
                # x[x<(1000-160)] = 1000-160
                # x[x>(1000+240)] = 1000+240
                # x = (x - x.min())/(x.max()-x.min())


                # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/check1006.tif', np.uint8(x * 255)) 
                # x = io.imread('{}/{}/DICOM_anon/volume.tif'.format(data_root, sample))
                # x = np.float32(x)/2500
                # x[x>1] = 1
                #
                D, H, W = x.shape
                D = int(D * d_resolution) #  
                H = int(H * h_resolution) # 
                W = int(W * w_resolution)  #  
                # we resample such that 1 pixel is 1 mm in x,y and z directiions
                base_grid = torch.zeros((1, D, H, W, 3))
                w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
                h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
                d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
                base_grid[:, :, :, :, 0] = w_points
                base_grid[:, :, :, :, 1] = h_points
                base_grid[:, :, :, :, 2] = d_points
                
                grid = base_grid.cuda()
                 
                
                x = torch.from_numpy(x).cuda()
                x = F.grid_sample(x[None, None], grid, mode='bilinear', padding_mode='border')[0, 0]
                x = x.data.cpu().numpy() 
                #----
                 
                x = np.float32(x) 
                mean_x = np.mean(x)
                std_x = np.std(x)

                D, H, W = x.shape
                center_z, center_y, center_x = D // 2, H // 2, W // 2
                D, H, W = pad_shape
                x = crop(x, (D, H, W), (center_z, center_y, center_x))  
 
                # io.imsave('{}/{}/DICOM_anon/volume_resampled_2.tif'.format(data_root, sample), np.uint16(x))
                 
                x = (x - mean_x)/std_x
                x = torch.from_numpy(x)
                inputs += [x]
                 
                #----
 
                y = [] 
                images_path = [dir for dir in os.listdir('{}/{}/Ground'.format(data_root, sample)) if 'png' in dir]
                for image_path in images_path:
                    file = io.imread('{}/{}/Ground/{}'.format(data_root, sample, image_path))
                    y += [file]  
                 
                y = np.array(y) 
                y = np.int64(y) 

                y = torch.from_numpy(y).cuda()
                y = F.grid_sample(y[None, None].float(), grid, mode='nearest', padding_mode='border')[0, 0]
                y = y.data.cpu().numpy()

                 
               
                y = np.int64(y)
                y = crop(y, (D, H, W), (center_z, center_y, center_x))  
                 
                # io.imsave('{}/{}/Ground/labels_resampled_2.tif'.format(data_root, sample), np.uint8(y))
                 
                y = torch.from_numpy(y/255) 
                 

                

                # y = np.uint8(y.data.cpu().numpy())
                # y = np.sum(y, axis=1)
                # y = np.sum(y, axis=1)
                # se = np.where(y>0)
                # embed()
                # print('{} {} {}'.format(sample, y.shape[0], se[0][-1]-se[0][0]))
                # print('{} {} {} {} {} {}'.format(sample, y.shape[0], y.shape[1], y.shape[2], se[0][0], se[0][-1]))
                labels += [y]

        # raise Exception()

        # inputs = []
        # labels = []
        # for sample in samples:
        
        #     if 'pickle' not in sample:
        #         print(sample)
        #         x = io.imread('{}/{}/DICOM_anon/volume_resampled_2.tif'.format(data_root, sample))

                
                
        #         inputs += [x]
        #         # print(sample)
        #         # print(x.shape)
        
        
        #         y = io.imread('{}/{}/Ground/labels_resampled_2.tif'.format(data_root, sample))
        #         y = np.int64(y/255)
        #         y = crop(y, (D, H, W), (center_z, center_y, center_x))  
        #         y = torch.from_numpy(y)
        #         labels += [y]
        # raise Exception()
        # print('loaded')
        # fix shuffle
        np.random.seed(1)
        perm = np.random.permutation(len(inputs))
        tr_length = cfg.training_set_size
        counts = [perm[:tr_length], perm[len(inputs)//2:]]
        # counts = [perm[:tr_length], perm[16:]]


        data = {}
        down_sample_shape = cfg.hint_patch_shape

        input_shape = x.shape
        scale_factor = (np.max(down_sample_shape)/np.max(input_shape))

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):

            samples = []
            print(i)
            print('--')

            for j in counts[i]:
                print(j)
                x = inputs[j]
                y = labels[j]

                x = F.interpolate(x[None, None], scale_factor=scale_factor, mode='trilinear')[0, 0]
                y = F.interpolate(y[None, None].float(), scale_factor=scale_factor, mode='nearest')[0, 0].long()

                new_samples = sample_to_sample_plus([Sample(x, y)], cfg, datamode) 
                samples.append(new_samples[0])
                # print('A BREAK IS HERE!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                # break 

            with open(data_root + '/pre_loaded_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = ChaosDataset(samples, cfg, datamode)
        print('-end-')
        data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)
        data[DataModes.VALIDATION] = data[DataModes.TESTING]
        # raise Exception()
        return data

    # def evaluate(self, target, pred, cfg):
    #     results = {}
    #
    #
    #     if target.voxel is not None:
    #         val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
    #         results['jaccard'] = val_jaccard
    #
    #     if target.mesh is not None:
    #         target_points = target.points
    #         pred_points = pred.mesh
    #         val_chamfer_weighted_symmetric = np.zeros(len(target_points))
    #
    #         for i in range(len(target_points)):
    #             val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])
    #
    #         results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric
    #
    #     return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        new_value = new_value[DataModes.TESTING][key]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.TESTING][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False

        # if 'chamfer_weighted_symmetric' in new_value[DataModes.TESTING]:
        #     key = 'chamfer_weighted_symmetric'
        #     new_value = new_value[DataModes.TESTING][key]

        #     if best_so_far is None:
        #         return True
        #     else:
        #         best_so_far = best_so_far[DataModes.TESTING][key]
        #         return True if np.mean(new_value) < np.mean(best_so_far) else False

        # elif 'jaccard' in new_value[DataModes.TESTING]:
        #     key = 'jaccard'
        #     new_value = new_value[DataModes.TESTING][key]

        #     if best_so_far is None:
        #         return True
        #     else:
        #         best_so_far = best_so_far[DataModes.TESTING][key]
        #         return True if np.mean(new_value) > np.mean(best_so_far) else False

# D = 6
# K = torch.zeros(1,1,D,D,D).cuda().float()
# K[0,0,D//2-1:D//2+1,D//2-1:D//2+1,D//2-1:D//2+1] = 1 

# x = F.conv3d(x_super_res[None, None].float(), K, bias=None, stride=D )[0, 0].float()/8
# y = (F.conv3d(y_super_res[None, None].float(), K, bias=None, stride=D )[0,0]>0).long()


# DD = 8
# a = (torch.rand(1,1,DD,DD).cuda() > 0.5).float()
# a = torch.ones(1,1,DD,DD).cuda().float()

# D = 4
# K = torch.zeros(1,1,D,D).cuda().float()
# K[0,0,D//2-1:D//2+1,D//2-1:D//2+1] = 1 

# a[0,0,:2] = 0
# a[0,0,:,:2] = 0 
# a[0,0,6:] = 0
# a[0,0,:,6:] = 0  
# F.interpolate(a, (2,2), mode='nearest') 
# F.conv2d(a.float(), K, bias=None, stride=4 )[0,0]

# D = 6
# K = torch.zeros(1,1,D,D,D).cuda().float()
# K[0,0,D//2-1:D//2+1,D//2-1:D//2+1,D//2-1:D//2+1] = 1 



# DD = 8
# a = torch.rand(1,1,DD,DD).cuda()
# D = 4
# P = DD//D
# K = torch.zeros(1,1,D,D).cuda().float()
# K[0,0,1:3,1:3] = 1 

# b1 = F.interpolate(a,(P,P),mode='bilinear') 
# b2 = F.conv2d(a, K,bias=None, stride=D )/(4)
# print(torch.max(torch.abs(b1-b2)))