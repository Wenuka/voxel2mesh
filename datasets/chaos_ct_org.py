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

    def __init__(self, data, cfg, mode): 
        self.data = data 
        # self.data = [data[0]] # <<<<<<<<<<<<<<<

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]  
        return get_item(item, self.mode, self.cfg)
        # return get_item_3(x, y, atlas, self.mode, self.cfg, atlas_resolution, self.perm[idx])
        # return get_item(x, y, atlas, self.mode, self.cfg, atlas_resolution)

  

class Chaos(DatasetAndSupport):

    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.hint_patch_shape
        data_root = '/cvlabsrc1'+volume_suffix+'/cvlab/datasets_udaranga/datasets/3d/chaos/Train_Sets/CT'
        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/pre_loaded_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                samples = pickle.load(handle)
                # if datamode == DataModes.TRAINING:
                #     samples = [samples[0]]
                # if datamode == DataModes.TESTING:
                #     samples = samples[:5]
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = ChaosDataset(new_samples, cfg, datamode)
        data[DataModes.TRAINING_EXTENDED] = ChaosDataset(data[DataModes.TRAINING].data, cfg, DataModes.TRAINING_EXTENDED)
        data[DataModes.VALIDATION] = data[DataModes.TESTING] ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< switch to testing
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