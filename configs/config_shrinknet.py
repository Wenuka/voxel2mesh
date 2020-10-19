import numpy as np
import torch 
from utils.utils_mesh import read_obj

class Config():
    def __init__(self):
        super(Config, self).__init__()

class ConfigMore(Config):
    def __init__(self):
        super(ConfigMore, self).__init__()

        ''' Loss lambdas '''
        self.lamda_ce = None
        self.lamda_angle = None

        ''' Priors '''
        self.priors = None

        self.latent_size = None
        

def load_config(exp_id):
      
    cfg = ConfigMore()
    ''' Experiment '''
    cfg.experiment_idx = exp_id 
    cfg.trial_id = None
 

    ''' 
    **** Paths *****
    save_path: results will be saved at this location
    dataset_path: dataset must be stored here.

    Note: During the first run use load_data function. It will do the necessary preprocessing and save the files at the same location.
    After that, you can use quick_load_data function to load them. This function is called in main.py

    '''

    cfg.save_path = '/cvlabdata1/cvlab/datasets_wenuka/experiments/' 

 
    cfg.N = 162 # 642 #  162, 2562
    cfg.sphere_path='/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/mesh_templates/spheres/icosahedron_{}.obj'.format(cfg.N)
 
    sphere_vertices, sphere_faces = read_obj(cfg.sphere_path)
    sphere_vertices = torch.from_numpy(sphere_vertices).cuda().float()
    cfg.sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])
 
    cfg.sphere_faces = torch.from_numpy(sphere_faces).cuda().long()

    # example
    # cfg.dataset_path = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/chaos/Train_Sets/CT'
    # cfg.save_path = '/cvlabdata2/cvlab/datasets_udaranga/experiments/vmnet/'
    cfg.save_dir_prefix = 'Experiment_' # prefix for experiment folder
 
    cfg.name = 'voxel2mesh'
    # cfg.name = 'unet'
   

    ''' Dataset ''' 
    cfg.training_set_size = 10  

    # input should be cubic. Otherwise, input should be padded accordingly.
    cfg.patch_shape = (96, 96, 96) 
    

    cfg.ndims = 3
    cfg.augmentation_shift_range = 10

    ''' Model '''
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 1
    cfg.steps = 4

    # Only supports batch size 1 at the moment. 
    cfg.batch_size = 1 


    cfg.num_classes = 2
    cfg.batch_norm = True  
    cfg.graph_conv_layer_count = 4

  
    ''' Optimizer '''
    cfg.learning_rate = 1e-4

    ''' Training '''
    cfg.numb_of_itrs = 120000
    cfg.eval_every = 2000 # saves results to disk

    # ''' Rreporting '''
    # cfg.wab = True # use weight and biases for reporting

    ''' ** To Verify ** '''
    cfg.unpool_indices = [0, 1, 1, 1, 0] #Ex3-31,33 ## use this always
    # cfg.unpool_indices = [0, 1, 0, 1, 1] #EX3- 32
    # cfg.unpool_indices = False ## for unet just to check

    cfg.sparse_model ='else' # 'point_model' , 'base_plane_model' or anything else lead to fully_annotated based training
    return cfg