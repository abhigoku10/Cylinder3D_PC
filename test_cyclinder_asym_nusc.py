# -*- coding:utf-8 -*-
# author: abhigoku10
# @file: train_cylinder_asym.py


### Have to be tested 

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name,get_nuScenes_label_color
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import warnings

warnings.filterwarnings("ignore")


def main(args):
    # pytorch_device = torch.device('cuda:0')
    pytorch_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    test_dataloader_config = configs['test_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    test_batch_size = test_dataloader_config['batch_size']

    model_config = configs['model_params']
    test_hypers = configs['test_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = test_hypers['model_load_path']
    output_path=test_hypers['output_save_path']

    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)
    
    test_dataset_loader, val_dataset_loader = data_builder.build_valtest(dataset_config,
                                                                  test_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    #### Validation module####
   
    print('*'*80)
    print("Processing the Validation section of Nuscenes data")
    print('*'*80)   
    pbar = tqdm(total=len(val_dataset_loader))
    time.sleep(10)
    my_model.eval()
    hist_list = []

    with torch.no_grad():
        for i_iter_val, (val_vox_fea, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                val_dataset_loader):
            val_vox_fea_ten = val_vox_fea.to(pytorch_device)

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            pbar.update(1)
            
            iou = per_class_iu(sum(hist_list))
            print('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))
            val_miou = np.nanmean(iou) * 100
            del val_vox_label, val_grid, val_pt_fea, val_grid_ten


            print('Current val miou is %.3f ' % (val_miou))

    pbar.close()
     #### Testing  module#### 
    print('*'*80)
    print("Processing the Testing section of Nuscenes data")
    print('*'*80) 
    print("The length of the test dataset is {}".format(len(test_dataset_loader)))

    pbar = tqdm(total=len(test_dataset_loader))
    hist_list = []
    time_list = []

    with torch.no_grad():
        for i_iter_val, (_,test_vox_label,test_grid,test_pt_labs,test_pt_fea,test_index,filename) in enumerate(test_dataset_loader):
#             
            test_label_tensor = test_vox_label.type(torch.LongTensor).to(pytorch_device)
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                            test_pt_fea]
            test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]         
            predict_labels = my_model(test_pt_fea_ten, test_grid_ten,test_batch_size)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
           

            # write to label file
            for count,i_test_grid in enumerate(test_grid):
                test_pred_label = predict_labels[count,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]]

                test_pred_label = np.expand_dims(test_pred_label,axis=1)



                _,dir2 = filename[0].split('/LIDAR_TOP/',1)
                new_save_dir = output_path + '/sequence_nusc/' +dir2.replace('velodyne','predictions')[:-3]+'label'                
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise

                 ####need to add this module        
                # test_pred_label=get_nuScenes_label_color(dataset_config["label_mapping"],test_pred_label)
                test_pred_label = test_pred_label.astype(np.int64)
#          
                test_pred_label.tofile(new_save_dir)

            ##### To check the predicted results 
            for count, i_test_grid in enumerate(test_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                count, test_grid[count][:, 0], test_grid[count][:, 1],
                                test_grid[count][:, 2]], test_pt_labs[count],
                            unique_label))

            pbar.update(1)
        iou = per_class_iu(sum(hist_list))
        print('*'*80)
        print('Testing per class iou: ')
        print('*'*80)
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        test_miou = np.nanmean(iou) * 100
        print('Current test miou is %.3f ' % test_miou)
        print('Inference time per %d is %.4f seconds\n' %
        (test_batch_size,np.mean(time_list)))
    del test_vox_label, test_grid, test_pt_fea, test_grid_ten,test_index
    pbar.close()
  


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
