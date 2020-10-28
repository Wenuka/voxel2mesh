import logging
import torch
# from torch.utils.tensorboard import SummaryWriter
from utils.utils_common import DataModes
# import torch.optim as optim

import numpy as np
import time 
import wandb
# from IPython import embed
import time 
import wandb
logger = logging.getLogger(__name__)


 
class Trainer(object):

 
    def training_step(self, data, epoch):
        # Get the minibatch 
         
        self.optimizer.zero_grad()
        loss, log = self.net.loss(data, epoch) 
        loss.backward()
        self.optimizer.step()  
        # self.scheduler.step()
        # embed()

        return log

    def __init__(self, net, trainloader, optimizer, numb_of_itrs, eval_every, save_path, evaluator):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer 

        self.numb_of_itrs = numb_of_itrs
        self.eval_every = eval_every
        self.save_path = save_path 

        self.evaluator = evaluator 
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)




    def train(self, start_iteration=1):

        print("Start training...") 
 
        self.net = self.net.train()
        iteration = start_iteration 

        print_every = 10
        while (True):
 
            for itr, data in enumerate(self.trainloader): # for nueron dataset, 14 sets
  
                # training step 
                # loss = self.training_step(data, start_iteration)
                loss = self.training_step(data, iteration)

                if iteration % print_every == 0:
                    log_vals = {}
                    for key, value in loss.items():
                        log_vals[key] = value / print_every
                    log_vals['lr'] = self.optimizer.param_groups[0]['lr']
                    log_vals['iteration'] = iteration 
                    wandb.log(log_vals)



                iteration = iteration + 1 

                if iteration % self.eval_every == self.eval_every-1:  # print every K epochs
                    self.evaluator.evaluate(iteration)

            if iteration > self.numb_of_itrs:
                break
   

        logger.info("... end training!")
 