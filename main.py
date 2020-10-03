 
import os
GPU_index = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

  
import logging
import torch
import numpy as np
from train import Trainer
from evaluate import Evaluator  
# from data.chaos import Chaos
from datasets.CORTEX_EPFL.cortexepfl_mesh3 import CortexEpfl
from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import DataModes
import wandb
# from IPython import embed 
from utils.utils_common import mkdir

from configs.config_shrinknet import load_config



logger = logging.getLogger(__name__)

class ShrinkNetLoss():


    def __init__(self): 
        # self.lambda_chamfer = 3000
        # self.lambda_edge = 300
        # self.lambda_laplacian = 1500
        # self.lambda_move = 100

        # main losses
        self.lambda_chamfer = 1
        self.lambda_mse = 0
        self.lambda_ce = 1
        self.lambda_hausdorff = 0
        self.lambda_chamfer_refined = 1
 
        # regularizations
        self.lambda_laplacian = 1
        self.lambda_normal_consistency = 1
        self.lambda_mesh_edge = 1

        self.lambda_alpha = 0.25
        self.lambda_beta = 0.25
 
def init(cfg):

    save_path = cfg.save_path + cfg.save_dir_prefix + str(cfg.experiment_idx).zfill(3)
    
    mkdir(save_path) 
 
    trial_id = (len([dir for dir in os.listdir(save_path) if 'trial' in dir]) + 1) if cfg.trial_id is None else cfg.trial_id
    trial_save_path = save_path + '/trial_' + str(trial_id) 

    if not os.path.isdir(trial_save_path):
        mkdir(trial_save_path) 
        copytree(os.getcwd(), trial_save_path + '/source_code', ignore=ignore_patterns('*.git','*.txt','*.tif', '*.pkl', '*.off', '*.so', '*.json','*.jsonl','*.log','*.patch','*.yaml','wandb','run-*'))

  
    seed = trial_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation 

    return trial_save_path, trial_id

def main():
    exp_id = 2
    cfg = load_config(exp_id)

    if cfg.name == 'voxel2mesh':
        from models.voxel2mesh import Voxel2Mesh as network
        print ("using voxel2mesh model")
    elif cfg.name == 'unet':
        from models.unet import UNet as network
        print ("using unet model")
    else:
        print ("Error at selecting the model. Please correct at config file.")
        exit()

    # Initialize
    trial_path, trial_id = init(cfg) 
 
    print('Experiment ID: {}, Trial ID: {}'.format(cfg.experiment_idx, trial_id))

    print("Create network")
    classifier = network(cfg)
    classifier.cuda()
 
    wandb.init(name='Experiment_{}/trial_{}'.format(cfg.experiment_idx, trial_id), project="voxel2mesh", dir='/cvlabdata1/cvlab/datasets_wenuka/experiments/wandb')
 
    print("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=cfg.learning_rate)  
  
    print("Load data") 
    # data_obj = Chaos()
    data_obj = CortexEpfl()

    # During the first run use load_data function. It will do the necessary preprocessing and save the files to disk.
    # data = data_obj.pre_process_dataset(cfg, trial_id)
    # data = data_obj.quick_load_data(cfg, trial_id)
    data = data_obj.quick_load_data(cfg, trial_id)
    
    loader = DataLoader(data[DataModes.TRAINING], batch_size=classifier.config.batch_size, shuffle=True)
  
    print("Trainset length: {}".format(loader.__len__()))

    print("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_obj) 

    print("Initialize trainer")
    trainer = Trainer(classifier, loader, optimizer, cfg.numb_of_itrs, cfg.eval_every, trial_path, evaluator)

    if cfg.trial_id is not None:
        print("Loading pretrained network")
        save_path = trial_path + '/best_performance/model.pth'
        checkpoint = torch.load(save_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0


    trainer.train(start_iteration=epoch) 

    # To evaluate a pretrained model, uncomment line below and comment the line above
    evaluator.evaluate(epoch)

if __name__ == "__main__": 
    main()