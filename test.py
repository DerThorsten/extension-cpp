import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import inferno

import torch.nn as nn
from inferno.io.box.cifar import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.convolutional import ConvELU2D
from inferno.extensions.layers.reshape import Flatten

import os
import vigra
import numpy
import nifty.cgp
import nifty.segmentation
import nifty.ground_truth

from inferno.extensions.layers.convolutional import *

from model import *
from bsd_ds import *

from predictor import * 




def main():

    laptop = True

    class LossWrapper(nn.Module):
        def __init__(self):
            super(LossWrapper, self).__init__()
            #self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]), reduce=False)
            self.loss = torch.nn.BCELoss(reduce=False)
        def forward(self, input, target):
            gt, sizes = target
            input = torch.squeeze(input)
            gt = torch.squeeze(gt)
            w = (gt + 1)**2.75
            sizes = torch.squeeze(sizes.float())
            #print('it', input.size(), target.size())
            
            l =  self.loss(input, gt)
            l = l * w
            l = torch.sum(l * sizes) / torch.sum(sizes)

            print("l",l.item())
            return l



    # Fill these in:
    if laptop:
        LOG_DIRECTORY = '/home/tbeier/src/extension-cpp/log/'
        SAVE_DIRECTORY = '/home/tbeier/src/extension-cpp/savedir_new/'
        RES_DIRECTORY = '/home/tbeier/src/extension-cpp/danu/'
        pmap_root = "/home/tbeier/src/holy-edge/hed-data/out/"
        bsd_root = "/home/tbeier/datasets/BSR/BSDS500/"

    else:
        LOG_DIRECTORY = '/export/home/tbeier/src/extension-cpp/log/'
        SAVE_DIRECTORY = '/export/home/tbeier/src/extension-cpp/savedir_new/'
        RES_DIRECTORY = '/export/home/tbeier/src/extension-cpp/danu/'
        pmap_root = "/export/home/tbeier/bsd500_HED/"
        bsd_root = "/home/tbeier/dataset/BSR/BSDS500/"

    USE_CUDA = bool(1)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Define transforms (1)
    #transforms = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])



    #split = "train"


    bsd_test = Bsd500Sp(
        bsd_root=bsd_root, 
        pmap_root=pmap_root,
        split='test')

    model = ConvNet()#.double()

    #model.double()
    # Build trainer
    trainer = Trainer(model) \
        .build_criterion(LossWrapper) \
        .build_criterion(LossWrapper) \
        .build_optimizer('Adam') \
        .validate_every((4, 'epochs')) \
        .save_every((4, 'epochs')) \
        .save_to_directory(SAVE_DIRECTORY) \
        .set_max_num_epochs(100) \
        # .build_logger(TensorboardLogger(log_scalars_every='never',
        #                             log_images_every='never'), 
        #           log_directory=LOG_DIRECTORY)
    
    # Bind loaders
    #train_loader = torch.utils.data.DataLoader(dataset=bsd_train, num_workers=8)
    #val_loader = torch.utils.data.DataLoader(dataset=bsd_val, num_workers=8)

    #trainer \
    #  .bind_loader('train',    train_loader, num_inputs=7, num_targets=2) \
    #  .bind_loader('validate', val_loader, num_inputs=7, num_targets=2)



    #checkpoint = torch.load('savedir/checkpoint.pytorch')
    # args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    #model.load_state_dict(checkpoint['state_dict'])


    trainer.cuda()
    trainer.load(best=True, from_directory=SAVE_DIRECTORY, filename='best_checkpoint.pytorch')
    meval = trainer.model.eval()#.cpu()

    #model.load_state_dict(torch.load('savedir/checkpoint.pytorch'))
    acc_vi_ds = 0.0
    acc_ri_ds = 0.0
    count = 0


    for i in range(0, 200):


        predictor = Predictor(model=meval, ds=bsd_test, output_folder=RES_DIRECTORY)

        
        vi_img,ri_img = predictor.predict_mc(i)

        acc_vi_ds += vi_img
        acc_ri_ds += ri_img
        count += 1
        print("\n",i)
        print("IMG ",vi_img, ri_img)
        print("DS  ",acc_vi_ds/count,  acc_ri_ds/count)


if __name__ == '__main__':
    main()