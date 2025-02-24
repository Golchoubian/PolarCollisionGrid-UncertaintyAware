import argparse
import pickle
import os
import time 

import torch
from torch.autograd import Variable

from utils.DataLoader import DataLoader
from utils.grid import getSequenceGridMask, getGridMask, getSequenceGridMask_heterogeneous, getGridMask_heterogeneous
from utils.Interaction import getInteractionGridMask, getSequenceInteractionGridMask
from utils.helper import * # want to use its get_model()
from utils.helper import sample_gaussian_2d
from matplotlib import pyplot as plt
import pandas as pd
import logging
import sys

def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=6, # 6 for HBS
                            help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=6, # 6 for HBS
                            help='Predicted length of the trajectory')

    # cuda support
    parser.add_argument('--use_cuda', action="store_true", default=True,
                            help='Use GPU or not')      
    # number of iteration                  
    parser.add_argument('--iteration', type=int, default=200, # 200
                            help='Number of saved models (during training) for testing their performance here \
                                  (smallest test errror will be selected)')

    # ============================================================================================
    #       change the following three arguments according to the model you want to test
    # ============================================================================================

    # method selection. this have to match with the training method manually
    parser.add_argument('--method', type=int, default=4,
                            help='Method of lstm being used (1 = social lstm, 3 = vanilla lstm, 4 = collision grid')
    
    # Wether the model is trained with uncertainty aware loss or not
    parser.add_argument('--uncertainty_aware', type=bool, default=True) # True for UAW-PCG, False for PCG

    # Model to be loaded (saved model (the epoch #) during training
    # with the best performace according to previous invesigation on valiquation set)
    parser.add_argument('--epoch', type=int, default= 175, # PCG: 196, UAW-PCG: 175
                            help='Epoch of model to be loaded')
    
    # The number of samples to be generated for each test data, when reporting its performance
    parser.add_argument('--sample_size', type=int, default=1,
                            help='The number of sample trajectory that will be generated')
        

    sample_args = parser.parse_args()

    seq_length = sample_args.obs_length + sample_args.pred_length

    dataloader = DataLoader(1, seq_length, infer=True, filtering=True)
    dataloader.reset_batch_pointer()

    # Define the path for the config file for saved args
    prefix = 'Store_Results/'
    save_directory_pre = os.path.join(prefix, 'model/')
    if sample_args.method == 1:
        save_directory = os.path.join(save_directory_pre, 'SocialLSTM/') 
    elif sample_args.method == 3:
        save_directory = os.path.join(save_directory_pre, 'VanillaLSTM/')
    elif sample_args.method == 4:
        if sample_args.uncertainty_aware:
            save_directory = os.path.join(save_directory_pre, 'UAW-PCG/')
        else:
            save_directory = os.path.join(save_directory_pre, 'PCG/')       
    else:
        raise ValueError('The selected method is not defined!')

    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    model_name = "LSTM"
    method_name = "SOCIALLSTM" # Attention: This name has not been changed for different models used. (ToDO later)
    save_tar_name = method_name+"_lstm_model_" 
    if saved_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    plot_directory = os.path.join(prefix, 'plot/')
    plot_test_file_directory = 'test'
    test_directory = os.path.join(plot_directory, plot_test_file_directory)

    log_file = os.path.join(test_directory, 'test_results.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    iteration_result = []
    iteration_total_error = []
    iteration_final_error = []
    iteration_MHD_error = []
    iteration_speed_error = []
    iteration_heading_error = []
    iteration_collision_percent = []
    iteration_collision_percent_pedped = []
    iteration_collision_percent_pedveh = []
    iteration_NLL_list = []
    iteration_ESV_sigma1 = []
    iteration_ESV_sigma2 = []
    iteration_ESV_sigma3 = []
    
    smallest_err = 100000
    smallest_err_iter_num = -1

    # Use "range(0, sample_args.iteration):" when willing to find the best trained model
    # in that case uncomment line 131 and comment line 132
    # This iteration is for testing the results for different stages of the trained model
    # (the stored paramters of the model at different iterations)
    start_iteration = 0
    for iteration in [0]: # range(start_iteration, sample_args.iteration): 
        # Initialize net
        net = get_model(sample_args.method, saved_args, True)

        if sample_args.use_cuda:        
            net = net.cuda()  

        # Get the checkpoint path for loading the trained model
        # checkpoint_path = os.path.join(save_directory, save_tar_name+str(iteration)+'.tar')
        checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        else:
            raise ValueError('The seleted model checkpoint does not exists in the specified directory!')

        # Initialzing some of the parameters 
        
        results = []
        
        # Variable to maintain total error
        total_error = 0
        final_error = 0
        MHD_error = 0
        speed_error = 0
        heading_error = 0

        num_collision_homo = 0
        all_num_cases_homo = 0
        num_collision_hetero = 0
        all_num_cases_hetero = 0

        NLL_list = []
        ESV_sigma1 = 0
        ESV_sigma2 = 0
        ESV_sigma3 = 0
        data_point_num = 0


        x_WholeBatch, numPedsList_WholeBatch, PedsList_WholeBatch, x_veh_WholeBatch, numVehsList_WholeBatch,\
            VehsList_WholeBatch, grids_WholeBatch, grids_veh_WholeBatch, grids_TTC_WholeBatch, grids_TTC_veh_WholeBatch \
                 = dataloader.batch_creater(False, sample_args.method, suffle=False)

        None_count = 0
        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get data
            # x, y, d , numPedsList, PedsList, x_veh, numVehsList, VehsList = dataloader.next_batch()
            x, numPedsList, PedsList = x_WholeBatch[batch], numPedsList_WholeBatch[batch], PedsList_WholeBatch[batch]
            x_veh, numVehsList, VehsList = x_veh_WholeBatch[batch], numVehsList_WholeBatch[batch], VehsList_WholeBatch[batch]

            x_seq, numPedsList_seq, PedsList_seq = x[0], numPedsList[0], PedsList[0] 
            x_seq_veh , numVehsList_seq, VehsList_seq = x_veh[0], numVehsList[0], VehsList[0]

            
            # dense vector creation
            x_seq, lookup_seq, mask = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            x_seq_veh, lookup_seq_veh, _ = dataloader.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)

            # will be used for error calculation
            orig_x_seq = x_seq.clone() 
            orig_x_seq_veh = x_seq_veh.clone()

            # grid mask calculation
            if  saved_args.method == 1: # social lstm   
                grid_seq = getSequenceGridMask(x_seq, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size,
                                                saved_args.use_cuda, lookup_seq) 
            
            elif saved_args.method == 4: # collision grid
                grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq, PedsList_seq,
                                                                        saved_args.TTC, saved_args.D_min, saved_args.num_sector,
                                                                        False, lookup_seq, lookup_seq)
                grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq_veh,
                                                                                        VehsList_seq, saved_args.TTC_veh,
                                                                                        saved_args.D_min_veh, saved_args.num_sector,
                                                                                        False, lookup_seq, lookup_seq_veh,
                                                                                        is_heterogeneous=True, is_occupancy=False)
            
            elif saved_args.method == 3:
                grid_seq = None

            
            # vectorize datapoints
            x_seq, first_values_dict = position_change_seq(x_seq, PedsList_seq, lookup_seq)

            # create the covaraince matrix using kalman filter and add it to x_seq
            GT_filtered_disp, GT_covariance = KF_covariance_generator(x_seq, mask, dataloader.timestamp)



            # add the covariances to x_seq
            covariance_flat = GT_covariance.reshape(GT_covariance.shape[0], GT_covariance.shape[1], 4)
            # x_seq up to here: [x, y, vx, vy, timestamp, ax, ay, speed_change, heading_change]
            x_seq = torch.cat((x_seq, covariance_flat), dim=2) 
            # x_seq: [x, y, vx, vy, timestamp, ax, ay, speed_change, heading_change, cov11, cov12, cov21, cov22]

            # adding the GT covariances to the x_orgi_seq for plotting them later on in the visualization
            orig_x_seq = torch.cat((orig_x_seq, covariance_flat), dim=2)
         
            if sample_args.use_cuda:
                x_seq = x_seq.cuda()
                if saved_args.method == 4:
                    x_seq_veh = x_seq_veh.cuda()


            sample_error = []
            sample_final_error = []
            sample_MHD_error = []
            sample_speed_error = []
            sample_heading_error = []
            sample_ret_x_seq = []
            sample_dist_param_seq = []
            sample_num_coll_hetero = []
            sample_num_coll_homo = []
            sample_all_num_cases_homo = []
            sample_all_num_cases_hetero = []
            sample_NLL = []
            sample_NLL_loss = []
            sample_ESV_sigma1 = []
            sample_ESV_sigma2 = []
            sample_ESV_sigma3 = []


            for sample_num in range(sample_args.sample_size):


                # The sample function
                if saved_args.method == 3: # vanilla lstm
                    # Extract the observed part of the trajectories
                    obs_traj, obs_PedsList_seq = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length]
                    ret_x_seq, dist_param_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq,
                                                        saved_args, dataloader, lookup_seq, numPedsList_seq, saved_args.gru,
                                                        first_values_dict, orig_x_seq)

                elif saved_args.method == 4: # collision grid
                    # Extract the observed part of the trajectories
                    obs_traj, obs_PedsList_seq, obs_grid, obs_grid_TTC = x_seq[:sample_args.obs_length], \
                                                                        PedsList_seq[:sample_args.obs_length], \
                                                                        grid_seq[:sample_args.obs_length], \
                                                                        grid_TTC_seq[:sample_args.obs_length]
                    obs_grid_veh_in_ped, obs_grid_TTC_veh = grid_seq_veh_in_ped[:sample_args.obs_length], \
                                                            grid_TTC_veh_seq[:sample_args.obs_length]
                  

                    ret_x_seq, dist_param_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq,
                                                        PedsList_seq, saved_args, dataloader, lookup_seq,
                                                        numPedsList_seq, saved_args.gru, first_values_dict,
                                                        orig_x_seq, obs_grid, x_seq_veh, VehsList_seq, 
                                                        lookup_seq_veh, obs_grid_veh_in_ped,
                                                        obs_grid_TTC, obs_grid_TTC_veh)

                elif saved_args.method == 1: # soial lstm
                    # Extract the observed part of the trajectories
                    obs_traj, obs_PedsList_seq, obs_grid = x_seq[:sample_args.obs_length], \
                                                           PedsList_seq[:sample_args.obs_length], \
                                                           grid_seq[:sample_args.obs_length]
                 
                    ret_x_seq, dist_param_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq,
                                                        PedsList_seq, saved_args, dataloader, lookup_seq,
                                                        numPedsList_seq, saved_args.gru, first_values_dict,
                                                        orig_x_seq, obs_grid, x_seq_veh, VehsList_seq, 
                                                        lookup_seq_veh, None, None, None)


                ret_x_seq = revert_postion_change_seq2(ret_x_seq.cpu(), PedsList_seq, lookup_seq, first_values_dict,
                                                        orig_x_seq, sample_args.obs_length, infer=True)
                dist_param_seq[:,:,0:2] = revert_postion_change_seq2(dist_param_seq[:,:,0:2].cpu(), PedsList_seq, 
                                                                     lookup_seq, first_values_dict, orig_x_seq, 
                                                                     sample_args.obs_length, infer=True)

                # Getting the agent_ids of those present in the observation section. 
                # error should be calculated only for those agents that their data exists in the observation part
                PedsList_obs = sum(PedsList_seq[:sample_args.obs_length], []) # contains duplicat but does not make any problem


                sample_error_batch = get_mean_error(ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                    orig_x_seq[sample_args.obs_length:,:,:2].data,
                                                    PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
                sample_final_error_batch = get_final_error(ret_x_seq[sample_args.obs_length:,:,:2].data, 
                                                           orig_x_seq[sample_args.obs_length:,:,:2].data, 
                                                           PedsList_seq[sample_args.obs_length:],PedsList_obs, lookup_seq)
                sample_MHD_error_batch = get_hausdorff_distance(ret_x_seq[sample_args.obs_length:,:,:2].data, 
                                                                orig_x_seq[sample_args.obs_length:,:,:2].data, 
                                                                PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
                sample_speed_error_batch, sample_heading_error_batch = get_velocity_errors(ret_x_seq[sample_args.obs_length:,:,2:4].data,
                                                                                            orig_x_seq[sample_args.obs_length:,:,2:4].data,
                                                                                            PedsList_seq[sample_args.obs_length:], PedsList_obs,
                                                                                            False, lookup_seq)
                sample_num_coll_homo_batch, sample_all_num_cases_homo_batch = get_num_collisions_homo(ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                                       PedsList_seq[sample_args.obs_length:], 
                                                                                                       PedsList_obs, False, lookup_seq,
                                                                                                        saved_args.D_min)
                sample_num_coll_hetero_batch, sample_all_num_cases_hetero_batch = get_num_collisions_hetero(   
                                                                                                ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                                PedsList_seq[sample_args.obs_length:], 
                                                                                                PedsList_obs, False, lookup_seq,
                                                                                                x_seq_veh[sample_args.obs_length:,:,:2].cpu().data,
                                                                                                VehsList_seq[sample_args.obs_length:], 
                                                                                                lookup_seq_veh, saved_args.D_min_veh)
                sample_NLL_list_batch, sample_NLL_loss_batch = get_mean_NLL(dist_param_seq[sample_args.obs_length:,:,:].data,
                                                                            orig_x_seq[sample_args.obs_length:,:,:2].data,
                                                                            PedsList_seq[sample_args.obs_length:], 
                                                                            PedsList_obs, False, lookup_seq)
                
                sample_ESV_sigma1_batch, sample_ESV_sigma2_batch, sample_ESV_sigma3_batch, counter = Delta_Empirical_Sigma_Value(
                                                                                                dist_param_seq[sample_args.obs_length:,:,:].data,
                                                                                                orig_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                                PedsList_seq[sample_args.obs_length:], PedsList_obs,
                                                                                                False, lookup_seq)


                sample_error.append(sample_error_batch)
                sample_final_error.append(sample_final_error_batch)
                sample_MHD_error.append(sample_MHD_error_batch)
                sample_speed_error.append(sample_speed_error_batch)
                sample_heading_error.append(sample_heading_error_batch)
                sample_ret_x_seq.append((ret_x_seq.clone()).data.cpu().numpy())
                sample_dist_param_seq.append((dist_param_seq.clone()).data.cpu().numpy())
                sample_num_coll_homo.append(sample_num_coll_homo_batch)
                sample_all_num_cases_homo.append(sample_all_num_cases_homo_batch)
                sample_num_coll_hetero.append(sample_num_coll_hetero_batch)
                sample_all_num_cases_hetero.append(sample_all_num_cases_hetero_batch)
                sample_NLL.append(sample_NLL_list_batch)
                sample_NLL_loss.append(sample_NLL_loss_batch)
                sample_ESV_sigma1.append(sample_ESV_sigma1_batch)
                sample_ESV_sigma2.append(sample_ESV_sigma2_batch)
                sample_ESV_sigma3.append(sample_ESV_sigma3_batch)

                

            # Deciding the best sample based on the average displacement error
            # We don't get the best sample of each individual. We get the best sample of all the agents in one batch (kind of average). This is a limitation
            min_ADE = min(sample_error)
            min_index = sample_error.index(min_ADE)
            # min_NLL = min(sample_NLL_loss)
            # min_index = sample_NLL_loss.index(min_NLL)

            total_error += sample_error[min_index] # or min_ADE
            final_error += sample_final_error[min_index]
            MHD_error += sample_MHD_error[min_index]
            speed_error += sample_speed_error[min_index]
            heading_error += sample_heading_error[min_index]
            num_collision_homo += sample_num_coll_homo[min_index]
            all_num_cases_homo += sample_all_num_cases_homo[min_index]
            num_collision_hetero += sample_num_coll_hetero[min_index]
            all_num_cases_hetero += sample_all_num_cases_hetero[min_index]
            NLL_list.extend(sample_NLL[min_index]) # getting the correct list of NLLs and adding that to the bigger list that contains different batches
            ESV_sigma1 += sample_ESV_sigma1[min_index]
            ESV_sigma2 += sample_ESV_sigma2[min_index]
            ESV_sigma3 += sample_ESV_sigma3[min_index]
            data_point_num += counter


            end = time.time()

            print('Current file : ', dataloader.get_file_name(0),' Processed trajectory number : ', batch+1,
                   'out of', dataloader.num_batches, 'trajectories in time', end - start)

            if sample_args.method == 3:
                results.append((orig_x_seq.data.cpu().numpy(), sample_ret_x_seq[min_index], PedsList_seq, lookup_seq , None, 
                                sample_args.obs_length, sample_dist_param_seq[min_index], orig_x_seq_veh.data.cpu().numpy(),
                                VehsList_seq, lookup_seq_veh, None, None))
            elif sample_args.method == 4:
                results.append((orig_x_seq.data.cpu().numpy(), sample_ret_x_seq[min_index], PedsList_seq, lookup_seq , None, 
                                sample_args.obs_length, sample_dist_param_seq[min_index], orig_x_seq_veh.data.cpu().numpy(), 
                                VehsList_seq, lookup_seq_veh, grid_seq, grid_seq_veh_in_ped))
            elif sample_args.method == 1:
                results.append((orig_x_seq.data.cpu().numpy(), sample_ret_x_seq[min_index], PedsList_seq, lookup_seq , None,
                                 sample_args.obs_length, sample_dist_param_seq[min_index], orig_x_seq_veh.data.cpu().numpy(), 
                                 VehsList_seq, lookup_seq_veh, grid_seq, None))

        ADE = (total_error.data.cpu()/ dataloader.num_batches).item()
        FDE = (final_error.data.cpu()/ dataloader.num_batches).item()
        MHD = (MHD_error.data.cpu()/ (dataloader.num_batches-None_count)).item()
        SE = (speed_error.data.cpu()/ dataloader.num_batches).item()
        HE = (heading_error.data.cpu()/ dataloader.num_batches).item()
        Collision = ((num_collision_homo+num_collision_hetero)/(all_num_cases_homo+all_num_cases_hetero))*100
        Collision_pedped = (num_collision_homo/all_num_cases_homo)*100
        Collision_pedveh = (num_collision_hetero/all_num_cases_hetero)*100
        sigma1 = (ESV_sigma1/data_point_num)-0.39
        sigma2 = (ESV_sigma2/data_point_num)-0.86
        sigma3 = (ESV_sigma3/data_point_num)-0.99

        iteration_result.append(results)
        iteration_total_error.append(ADE)
        iteration_final_error.append(FDE)
        iteration_MHD_error.append(MHD)
        iteration_speed_error.append(SE)
        iteration_heading_error.append(HE)
        iteration_collision_percent.append(Collision)
        iteration_collision_percent_pedped.append(Collision_pedped)
        iteration_collision_percent_pedveh.append(Collision_pedveh)
        iteration_NLL_list.append(NLL_list)
        iteration_ESV_sigma1.append(sigma1)
        iteration_ESV_sigma2.append(sigma2)
        iteration_ESV_sigma3.append(sigma3)
        
        print('Iteration:' ,iteration+1,' Total testing (prediction sequence) mean error of the model is ', ADE) 
        print('Iteration:' ,iteration+1,'Total testing final error of the model is ', FDE)
        print('Iteration:' ,iteration+1,'Total tresting (prediction sequence) hausdorff distance error of the model is ',
                                                                                     MHD)
        print('Iteration:' ,iteration+1,'Total tresting (prediction sequence) speed error of the model is ', SE)
        print('Iteration:' ,iteration+1,'Total tresting final heading error of the model is ', HE)
        print('Iteration:' ,iteration+1,'Overll percentage of collision of the model is ', Collision)
        print('Iteration:' ,iteration+1,'Percentage of collision between pedestrians of the model is ', Collision_pedped)
        print('Iteration:' ,iteration+1,'Percentage of collision between pedestrians and vehicles of the model is ',
                                                                                     Collision_pedveh)
        # calculate the mean and std for the NLL_list
        NLL_mean, NLL_std = calculate_mean_and_std(NLL_list)
        print('Iteration:' ,iteration+1,'NLL of the model is ', NLL_mean, 'with std of ', NLL_std)
        print('Iteration:' ,iteration+1,'ESV_sigma1 of the model is ', sigma1)
        print('Iteration:' ,iteration+1,'ESV_sigma2 of the model is ', sigma2)
        print('Iteration:' ,iteration+1,'ESV_sigma3 of the model is ', sigma3)

        # saving the informatino of all episodes on the eval dataset in a csv file
        df = pd.DataFrame({'model_num': [iteration], 'ADE': [ADE], 'FDE': [FDE], 'MHD': [MHD],
                            'speed_error': [SE], 'heading_error': [HE], 'collision_rate': [Collision],
                            'collision_rate_pedped': [Collision_pedped],
                            'collision_rate_pedveh': [Collision_pedveh],
                            'NLL_mean': [NLL_mean], 'NLL_std': [NLL_std],
                            'ESV_sigma1': [sigma1], 'ESV_sigma2': [sigma2], 'ESV_sigma3': [sigma3]})
        if os.path.exists(os.path.join(test_directory, 'eval.csv')):
            df.to_csv(os.path.join(test_directory, 'eval.csv'), mode='a', header=False, index=False)
        else:
            df.to_csv(os.path.join(test_directory, 'eval.csv'), mode='w', header=True, index=False)

        
        if total_error<smallest_err:
            print("**********************************************************")
            print('Best iteration has been changed. Previous best iteration: ', smallest_err_iter_num+1, 'Error: ',
                   smallest_err / dataloader.num_batches)
            print('New best iteration : ', iteration+1, 'Error: ',total_error / dataloader.num_batches)
            smallest_err_iter_num = iteration
            smallest_err = total_error

    best_model_list_index = smallest_err_iter_num-start_iteration
    dataloader.write_to_plot_file(iteration_result[best_model_list_index], 
                                  os.path.join(plot_directory, plot_test_file_directory)) 

    print("==================================================")
    print("==================================================")
    print("==================================================")
    print('Best final iteration : ', smallest_err_iter_num+1)
    print('ADE: ', iteration_total_error[best_model_list_index])
    print('FDE: ', iteration_final_error[best_model_list_index])
    print('MHD: ', iteration_MHD_error[best_model_list_index])
    print('Speed error: ',iteration_speed_error[best_model_list_index])
    print('Heading error: ', iteration_heading_error[best_model_list_index])
    print('Collision percentage: ', round(iteration_collision_percent[best_model_list_index], 4))
    print('Collision percentage (ped-ped): ', round(iteration_collision_percent_pedped[best_model_list_index], 4))
    print('Collision percentage (ped-veh): ', round(iteration_collision_percent_pedveh[best_model_list_index], 4))

    best_iter_NLL_list = iteration_NLL_list[best_model_list_index]
    # calculate the mean and std for this list 
    NLL_mean, NLL_std = calculate_mean_and_std(best_iter_NLL_list)
    print('NLL: ', NLL_mean, '±', NLL_std)
    print('ESV_sigma1: ', iteration_ESV_sigma1[best_model_list_index])
    print('ESV_sigma2: ', iteration_ESV_sigma2[best_model_list_index])
    print('ESV_sigma3: ', iteration_ESV_sigma3[best_model_list_index])

    # logging
    logging.info(
        'ADE: {:.3f}, FDE: {:.3f}, MHD: {:.3f}, Speed_error: {:.3f}, '
        'Heading_error: {:.3f}'.
        format(iteration_total_error[best_model_list_index], 
               iteration_final_error[best_model_list_index], 
               iteration_MHD_error[best_model_list_index],
               iteration_speed_error[best_model_list_index],
               iteration_heading_error[best_model_list_index]))
    logging.info(
        'Collision: {:.3f}%, Collision_pedped: {:.3f}%, '
        'Collision_pedveh: {:.3f}%'.
        format(iteration_collision_percent[best_model_list_index],
               iteration_collision_percent_pedped[best_model_list_index],
               iteration_collision_percent_pedveh[best_model_list_index]))
    logging.info(
        'NLL: {:.3f}+- {:.3f}, '
        'ESV_sigma1: {:.3f}, ESV_sigma2: {:.3f}, ESV_sigma3: {:.3f},'.
        format(NLL_mean, NLL_std, iteration_ESV_sigma1[best_model_list_index],
               iteration_ESV_sigma2[best_model_list_index],
               iteration_ESV_sigma3[best_model_list_index]))
    


def sample(x_seq, Pedlist, args, net, true_x_seq, true_Pedlist, saved_args, dataloader, look_up, num_pedlist, is_gru,
            first_values_dict, orig_x_seq, grid=None, x_seq_veh=None, Vehlist=None, look_up_veh=None,
            grid_veh_in_ped=None, grid_TTC=None, grid_TTC_veh=None):
    '''
    The sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    true_x_seq: True positions
    true_Pedlist: The true peds present in each frame
    saved_args: Training arguments
    '''

    # Number of peds in the sequence 
    # This will also include peds that do not exist in the observation length but come into scene in the prediction time interval
    numx_seq = len(look_up)   
    if look_up_veh is not None:
        numx_seq_veh = len(look_up_veh)
        
    with torch.no_grad():
        # Construct variables for hidden and cell states
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not is_gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, x_seq.shape[2]))
        dist_param_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, 5))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()
            # dist_param_seq = dist_param_seq.cuda()

        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1):
            if grid is None: 
               # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep,:,:].view(1, numx_seq, x_seq.shape[2]), hidden_states, cell_states, 
                                                          [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            elif (args.method == 4):
                # Do a forward prop 
                # We give the frames one by one as input. 
                grid_t = grid[tstep]
                grid_veh_in_ped_t = grid_veh_in_ped[tstep]
                grid_TTC_t = grid_TTC[tstep]
                grid_TTC_veh_t = grid_TTC_veh[tstep]
                if args.use_cuda:
                    grid_t = grid_t.cuda()
                    grid_veh_in_ped_t = grid_veh_in_ped_t.cuda()
                    grid_TTC_t = grid_TTC_t.cuda()
                    grid_TTC_veh_t = grid_TTC_veh_t.cuda()
                out_obs, hidden_states, cell_states = net(x_seq[tstep,:,:].view(1, numx_seq, x_seq.shape[2]), [grid_t],
                                                           hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader,
                                                            look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                            [grid_veh_in_ped_t], [Vehlist[tstep]], look_up_veh, [grid_TTC_t], [grid_TTC_veh_t])

            elif (args.method == 1):
                grid_t = grid[tstep]
                out_obs, hidden_states, cell_states = net(x_seq[tstep,:,:].view(1, numx_seq, x_seq.shape[2]), [grid_t.cpu()],
                                                           hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader,
                                                            look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]), None,
                                                            [Vehlist[tstep]], look_up_veh)

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_obs.cpu())

            # Storing the paramteres of the distriution for plotting
            dist_param_seq[tstep + 1, :, :] = out_obs.clone()

            # # Sample from the bivariate Gaussian
            # next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y

            # Assign the mean to the next state. 
            next_x_mean = mux
            next_y_mean = muy
            ret_x_seq[tstep + 1, :, 0] = next_x_mean
            ret_x_seq[tstep + 1, :, 1] = next_y_mean

    
        # Last seen grid
        if grid is not None: # not vanilla lstm
            prev_grid = grid[-1].clone()
            if (args.method == 4):
                prev_grid_veh_in_ped = grid_veh_in_ped[-1].clone()
                prev_TTC_grid = grid_TTC[-1].clone()
                prev_TTC_grid_veh = grid_TTC_veh[-1].clone()

        # constructing the speed change and deviation feautres for time step obs_length-1 
        # that is going ot be used in the next for loop and also calculating that for each new time step on the following for loop
        ret_x_seq[tstep + 1, :, 2:9] = x_seq[-1,:,2:9]
        ret_x_seq[tstep + 1, :, 9:13] = x_seq[-1,:,9:13] # covariances of the trjecotries generated by the Kalman filter

        last_observed_frame_prediction = ret_x_seq[tstep + 1, :, :2].clone()
        ret_x_seq[tstep + 1, :, :2] = x_seq[-1,:,:2] # storing the last GT observed frame here to ensure this is used in the next for loop and then 
        # storing the actual prediction in it after the forward network is run for the first step in the prediction length 


        timestamp = dataloader.timestamp

        # For the predicted part of the trajectory
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
            # Do a forward prop
            if grid is None: # vanilla lstm
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), hidden_states, 
                                                          cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            else:
                if (args.method == 4):
                    outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), [prev_grid],
                                                            hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]],
                                                            dataloader, look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]), 
                                                            [prev_grid_veh_in_ped], [Vehlist[tstep]], look_up_veh,  [prev_TTC_grid],
                                                            [prev_TTC_grid_veh])
                elif (args.method == 1):
                     outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), [prev_grid.cpu()],
                                                            hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]],
                                                            dataloader, look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                            None, [Vehlist[tstep]], look_up_veh)
            if tstep == args.obs_length-1: 
                # storing the actual prediction in the last observed frame position
                ret_x_seq[args.obs_length-1, :, :2] = last_observed_frame_prediction.clone()
      
            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs.cpu())

            # Storing the paramteres of the distriution
            dist_param_seq[tstep + 1, :, :] = outputs.clone() 

            # # Sample from the bivariate Gaussian
            # next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)
            # # Store the predicted position
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y

            # Assign the mean to the next state. 
            next_x_mean = mux
            next_y_mean = muy
            ret_x_seq[tstep + 1, :, 0] = next_x_mean
            ret_x_seq[tstep + 1, :, 1] = next_y_mean


            # Preparing a ret_x_seq that is covnerted back to the original frame by adding the first position.
            # This will be used for grid calculation
            ret_x_seq_convert = ret_x_seq.clone() 
            ret_x_seq_convert = revert_postion_change_seq2(ret_x_seq_convert.cpu(), true_Pedlist, look_up, first_values_dict,
                                                            orig_x_seq, saved_args.obs_length, infer=True)


            ret_x_seq_convert[tstep + 1, :, 2] = (ret_x_seq_convert[tstep + 1, :, 0] - ret_x_seq_convert[tstep, :, 0]) / timestamp # vx 
            ret_x_seq_convert[tstep + 1, :, 3] = (ret_x_seq_convert[tstep + 1, :, 1] - ret_x_seq_convert[tstep, :, 1]) / timestamp # vy
            # updating the velocity data in ret_x_seq accordingly
            ret_x_seq[tstep + 1, :, 2] = ret_x_seq_convert[tstep + 1, :, 2].clone()
            ret_x_seq[tstep + 1, :, 3] = ret_x_seq_convert[tstep + 1, :, 3].clone()

            
            # claculating the rest of the features that will be used in the interaction tensor (speed change and deviation) 
            ret_x_seq_convert[tstep + 1, :, 5] = (ret_x_seq_convert[tstep + 1, :, 2] - ret_x_seq_convert[tstep, :, 2]) / timestamp # ax
            ret_x_seq_convert[tstep + 1, :, 6] = (ret_x_seq_convert[tstep + 1, :, 3] - ret_x_seq_convert[tstep, :, 3]) / timestamp # ay
            speed_next_t = torch.pow((torch.pow(ret_x_seq_convert[tstep + 1, :, 2], 2) + torch.pow(ret_x_seq_convert[tstep + 1, :, 3], 2)),0.5)
            speed_t = torch.pow((torch.pow(ret_x_seq_convert[tstep, :, 2], 2) + torch.pow(ret_x_seq_convert[tstep, :, 3], 2)),0.5)
            ret_x_seq[tstep + 1, :, 7] = speed_next_t - speed_t # speed difference
            dot_vel = torch.mul(ret_x_seq_convert[tstep, :, 2], ret_x_seq_convert[tstep+1, :, 2]) + \
                        torch.mul(ret_x_seq_convert[tstep, :, 3], ret_x_seq_convert[tstep+1, :, 3])
            det_vel = torch.mul(ret_x_seq_convert[tstep, :, 2],  ret_x_seq_convert[tstep+1, :, 3]) - \
                        torch.mul(ret_x_seq_convert[tstep, :, 3], ret_x_seq_convert[tstep+1, :, 2])
            ret_x_seq[tstep + 1, :, 8] = torch.atan2(det_vel,dot_vel) * 180/np.pi # deviation angel

         
            scaled_param_dist = torch.stack((mux, muy, sx, sy, corr),2) 
            cov = cov_mat_generation(scaled_param_dist)
            ret_x_seq[tstep + 1, :, 9:13] = cov.reshape(cov.shape[0], cov.shape[1], 4).squeeze(0) # covariances of the trjecotries generated by the predictor

            # List of x_seq at the last time-step (assuming they exist until the end)
            true_Pedlist[tstep+1] = [int(_x_seq) for _x_seq in true_Pedlist[tstep+1]]
            next_ped_list = true_Pedlist[tstep+1].copy()
            converted_pedlist = [look_up[_x_seq] for _x_seq in next_ped_list]
            list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

            # Get their predicted positions
            current_x_seq = torch.index_select(ret_x_seq_convert[tstep+1], 0, list_of_x_seq)

            if grid is not None: # not vanilla lstm

                if args.method == 4:
                    Vehlist[tstep+1] = [int(_x_seq_veh) for _x_seq_veh in Vehlist[tstep+1]]
                    next_veh_list = Vehlist[tstep+1].copy()
                    converted_vehlist = [look_up_veh[_x_seq_veh] for _x_seq_veh in next_veh_list]
                    list_of_x_seq_veh = Variable(torch.LongTensor(converted_vehlist))
                    if args.use_cuda:
                        list_of_x_seq_veh = list_of_x_seq_veh.cuda()
                    current_x_seq_veh = torch.index_select(x_seq_veh[tstep+1], 0, list_of_x_seq_veh)
          
            
                if  args.method == 1: #social lstm 
                    prev_grid = getGridMask(current_x_seq.data.cpu(), len(true_Pedlist[tstep+1]),
                                            saved_args.neighborhood_size, saved_args.grid_size) 
                  
                elif args.method == 4: # Collision grid
                    prev_grid, prev_TTC_grid = getInteractionGridMask(current_x_seq.data.cpu(), current_x_seq.data.cpu(), saved_args.TTC,
                                                                       saved_args.D_min, saved_args.num_sector)
                    prev_grid_veh_in_ped, prev_TTC_grid_veh = getInteractionGridMask(current_x_seq.data.cpu(),  current_x_seq_veh.data.cpu(),
                                                                                      saved_args.TTC_veh, saved_args.D_min_veh, 
                                                                                      saved_args.num_sector, is_heterogeneous = True,
                                                                                      is_occupancy = False)


                prev_grid = Variable(torch.from_numpy(prev_grid).float())
              
                if args.method == 4:
                    prev_grid_veh_in_ped = Variable(torch.from_numpy(prev_grid_veh_in_ped).float())
                    prev_TTC_grid = Variable(torch.from_numpy(prev_TTC_grid).float())
                    prev_TTC_grid_veh = Variable(torch.from_numpy(prev_TTC_grid_veh).float())
                    if args.use_cuda:
                        prev_grid = prev_grid.cuda()
                        prev_grid_veh_in_ped = prev_grid_veh_in_ped.cuda()
                        # if args.method == 4:
                        prev_TTC_grid = prev_TTC_grid.cuda()
                        prev_TTC_grid_veh = prev_TTC_grid_veh.cuda()


        return ret_x_seq, dist_param_seq



if __name__ == '__main__':
    main()


    

