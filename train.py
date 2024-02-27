import torch 
from DataLoader import DataLoader
from helper import *
from grid import getSequenceGridMask, getSequenceGridMask_heterogeneous 
from Interaction import getInteractionGridMask, getSequenceInteractionGridMask
from torch.autograd import Variable
import time
import argparse
import os
import pickle
from visualization import Loss_Plot, plot_bivariate_gaussian3
from matplotlib import pyplot as plt
import random

'''
## Acknowledgements
This project is builds on top of the codebase from [social-lstm](https://github.com/quancore/social-lstm),
developed by "quancore" as a pytorch implementation of the Social LSTM model proposed by Alahi et al.
'''


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2) # if adding the covariance matrix to the input, the input size changes from 2 to 6
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10, # 
                        help='minibatch size')
    # Length of sequence to be considered
    parser.add_argument('--seq_length', type=int, default=12, # 12 for HBS (obs: 6, pred: 6)
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=6,
                        help='prediction length')
    parser.add_argument('--obs_length', type=int, default=6,
                        help='Observed length of the trajectory')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=200, 
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now.
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')

    # Dimension of the embeddings parameter for actions
    parser.add_argument('--embedding_size_action', type=int, default=32,
                        help='Embedding dimension for the actions')

    # For the SocialLSTM:
    # Size of neighborhood to be considered parameter # 
    parser.add_argument('--neighborhood_size', type=int, default=8,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')


    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # drive option
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--store_grid', action="store_true", default=True, # !!!!!!!!!!!!!!!!!!!!!!!!! 
                        help='Whether store grids and use further epoch')

    # Size of neighborhood for vehilces in pedestrians grid 
    parser.add_argument('--neighborhood_size_veh_in_ped', type=int, default=64,
                        help='Neighborhood size to be considered for social grid (the grid that considers vehicles)')
    # Size of the social grid parameter for vehilces in pedestrians grid
    parser.add_argument('--grid_size_veh_in_ped', type=int, default=4,
                        help='Grid size of the social grid (the grid that considers vehicles)')

    # The lateral size of the social grid, the number of divisions of the circle around the agent for specifying the approach angle
    parser.add_argument('--num_sector', type=int, default=8, 
                        help='The number of circle division for distinguishing approach angle')
    # Minimum time to collisions to be considered, the num of TTC is the radial size of the social grid mask
    parser.add_argument('--TTC', type=int, default=[9], # [10]
                        help='Minimum time to collisions to be considerd for the social grid')
    # Minimum acceptalbe distance between two pedestrians
    parser.add_argument('--D_min', type=int, default=0.7, 
                        help='Minimum distance for which the TTC is calculated')
    # Minimum time to collisions to be considered for ped-veh interaction, the num of TTC is the radial size of the social grid mask of veh in ped
    parser.add_argument('--TTC_veh', type=int, default=[8],
                        help='Minimum time to collisions to be considerd for the social grid')
    # Minimum acceptalbe distance between a pedstrian and a vehicle
    parser.add_argument('--D_min_veh', type=int, default=1.0,
                        help='Minimum distance for which the TTC is calculated')
    # method selection
    parser.add_argument('--method', type=int, default=4,
                            help='Method of lstm will be used (1 = social lstm, 3 = vanilla lstm, 4 = collision grid)') 
    # Wether to train the model with uncertainty aware loss or not
    parser.add_argument('--uncertainty_aware', type=bool, default=True) # True for UAW-PCG, False for PCG
     
    # resume training from an existing checkpoint or not
    parser.add_argument(
        '--resume', default=False, action='store_true')
    # if resume = True, load from the following checkpoint
    parser.add_argument(
        '--resume-model-path', default='Store_Results/model/saved_model/SOCIALLSTM_lstm_model_55.tar',
        help='path of weights for resume training')
    
    parser.add_argument('--teacher_forcing', action="store_true", default=False,
                        help='Whether to use teacher forcing or not during training') 
    # when not using teacher forcing, the model uses the last predicted state instead of true state for the prediction length part (same as test time)
    # when using teacher forcing, the model always uses the true state from the previous time step during training.
    args = parser.parse_args()

    train(args)



def train(args):


    model_name = "LSTM"
    method_name = "SOCIALLSTM" # Attention: This name has not been changed for different models used. (ToDO later)
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

     # Log directory
    prefix = 'Store_Results/'
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/') 

    # Logging files
    log_file_curve = open(os.path.join(log_directory,'log_curve.txt'), 'w+')

    # model directory
    save_directory = os.path.join(prefix, 'model/') 

    # Save the arguments in the config file
    with open(os.path.join(save_directory,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file 
    def checkpoint_path(x):
        return os.path.join(save_directory, save_tar_name+str(x)+'.tar')


    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, and of length args.seq_length
    dataloader = DataLoader(args.batch_size, args.seq_length, infer=False, filtering=True)


    # model creation
    net = get_model(args.method,args)
    if args.use_cuda:
        net = net.cuda()

    if args.resume:
        # Get the checkpoint path for loading the trained model
        model_checkpoint_path = args.resume_model_path
        if os.path.isfile(model_checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(model_checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        else:
            raise ValueError('The seleted model checkpoint does not exists in the specified directory!')

    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param) 
    # optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)


    if args.store_grid:
        print("////////////////////////////")
        print("Starting the off line grid caculation all at once")
        grid_cal_start = time.time()
        dataloader.grid_creation(args)
        grid_cal_end = time.time()
        print("grid calculation is finished")
        print("grid calculation time for all the data: {} seconds".format(grid_cal_end - grid_cal_start))  
        print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\")

    num_batch = 0


    start_train_loop = time.time()
    err_batch_list = []
    loss_batch_list = []
    train_batch_num_list = []
    loss_epoch_list = []
    err_epoch_list = []
    NLL_loss_batch_list = []
    uncertainty_loss_batch_list = []
    NLL_loss_epoch_list = []
    uncertainty_loss_epoch_list = []

    ax2 = None
    # fig2, ax2 = plt.subplots()
    # plt.ion()

    # Training
    for epoch in range(args.num_epochs):
        
        print('**************** Training epoch beginning ******************')
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0
        NLL_loss_epoch = 0
        uncertainty_loss_epoch = 0
        err_epoch = 0

        # changing the order of the sequence if shuffle in on
        x_WholeBatch, numPedsList_WholeBatch, PedsList_WholeBatch, x_veh_WholeBatch, numVehsList_WholeBatch, \
            VehsList_WholeBatch, grids_WholeBatch, grids_veh_WholeBatch, grids_TTC_WholeBatch, grids_TTC_veh_WholeBatch = \
                dataloader.batch_creater(args.store_grid, args.method, suffle=True)

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, numPedsList, PedsList = x_WholeBatch[batch], numPedsList_WholeBatch[batch], PedsList_WholeBatch[batch]
            x_veh, numVehsList, VehsList = x_veh_WholeBatch[batch], numVehsList_WholeBatch[batch], VehsList_WholeBatch[batch]
            if args.store_grid:
                grids_batch, grids_veh_batch = grids_WholeBatch[batch], grids_veh_WholeBatch[batch]
                if (args.method == 4):
                    grids_TTC_batch, grids_TTC_veh_batch = grids_TTC_WholeBatch[batch], grids_TTC_veh_WholeBatch[batch]


            loss_batch = 0
            err_batch = 0
            NLL_loss_batch = 0
            uncertainty_loss_batch = 0

            # Zero out gradients
            net.zero_grad()
            optimizer.zero_grad()

            # For each sequence
            for sequence in range(dataloader.batch_size):
              
                x_seq , numPedsList_seq, PedsList_seq = x[sequence], numPedsList[sequence], PedsList[sequence]
                x_seq_veh , numVehsList_seq, VehsList_seq = x_veh[sequence], numVehsList[sequence], VehsList[sequence]

                #dense vector creation
                x_seq, lookup_seq, mask = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq) 
                # order of featurs in x_seq: x, y, vx, vy, timestamp, ax, ay 
                x_seq_veh, lookup_seq_veh, mask_veh = dataloader.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)
              
                x_seq_orig = x_seq.clone()
                x_seq_veh_orig = x_seq_veh.clone()

                # # create thec covaraince matrix using kalman filter and add it to x_seq
                # GT_filtered_state, GT_covariance = KF_covariance_generator(x_seq, mask, dataloader.timestamp, plot_bivariate_gaussian3, ax2) # the last two arguments are for testing the KF with ploting the bivariate gaussian

                if args.store_grid:
                   grid_seq =  grids_batch[sequence]
                   grid_seq_veh_in_ped = grids_veh_batch[sequence]
                   if args.method == 4:
                    grid_TTC_seq = grids_TTC_batch[sequence]
                    grid_TTC_veh_seq = grids_TTC_veh_batch[sequence]
                
                else:
                    if args.method == 1: # Social LSTM
                        grid_seq = getSequenceGridMask(x_seq, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda, lookup_seq)
                        grid_seq_veh_in_ped = getSequenceGridMask_heterogeneous(x_seq, PedsList_seq, x_seq_veh, VehsList_seq,
                                                                                 args.neighborhood_size_veh_in_ped, args.grid_size_veh_in_ped, 
                                                                                 args.use_cuda, lookup_seq, lookup_seq_veh, False)
                    
                    elif args.method ==4: # CollisionGird
                        grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq, PedsList_seq, args.TTC,
                                                                                args.D_min, args.num_sector, args.use_cuda, 
                                                                                lookup_seq, lookup_seq)
                        grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq_veh, VehsList_seq,
                                                                                                args.TTC_veh, args.D_min_veh, args.num_sector,
                                                                                                args.use_cuda, lookup_seq, lookup_seq_veh,
                                                                                                is_heterogeneous=True, is_occupancy=False) 
                        
                x_seq, first_values_dict = position_change_seq(x_seq, PedsList_seq, lookup_seq)
                x_seq_veh, first_values_dict_veh = position_change_seq(x_seq_veh, VehsList_seq, lookup_seq_veh)
                velocity_change, _ = position_change_seq(x_seq[:,:,2:4], PedsList_seq, lookup_seq)
                x_seq[:,:,2:4] = velocity_change

                # create the covaraince matrix using kalman filter and add it to x_seq
                GT_filtered_disp, GT_covariance = KF_covariance_generator(x_seq, mask, dataloader.timestamp,
                                                                          plot_bivariate_gaussian3, ax2,
                                                                          x_seq_orig[:,:,:2], PedsList_seq, lookup_seq, args.use_cuda,
                                                                          first_values_dict, args.obs_length) 
                                                                        # the last arguments are for testing the KF with ploting the bivariate gaussian

                # add the covariances to x_seq
                covariance_flat = GT_covariance.reshape(GT_covariance.shape[0], GT_covariance.shape[1], 4)
                # x_seq up to here: [x, y, vx, vy, timestamp, ax, ay, speed_change, heading_change]
                x_seq = torch.cat((x_seq, covariance_flat), dim=2) 
                # x_seq: [x, y, vx, vy, timestamp, ax, ay, speed_change, heading_change, cov11, cov12, cov21, cov22]

                if args.use_cuda:                    
                    x_seq = x_seq.cuda()
                    x_seq_veh = x_seq_veh.cuda()
                    mask = mask.cuda()
                    GT_filtered_disp = GT_filtered_disp.cuda()
                    GT_covariance = GT_covariance.cuda()

                y_dist_mean = GT_filtered_disp[1:,:,:2]
                y_dis_cov = GT_covariance[1:,:,:2,:2]
                
                y_seq = x_seq[1:,:,:2]
                x_seq = x_seq[:-1,:,:]
                numPedsList_seq = numPedsList_seq[:-1]
             
                y_seq_veh = x_seq_veh[1:,:,:2]
                x_seq_veh = x_seq_veh[:,:,:] # x_seq_veh[:-1,:,:]
                numVehsList_seq = numVehsList_seq[:-1]
               

                if args.method != 3: # not Vanilla LSTM 
                    grid_seq_plot = grid_seq[1:]
                    grid_seq_veh_plot = grid_seq_veh_in_ped[1:]

                    grid_seq = grid_seq[:-1]
                    grid_seq_veh_in_ped = grid_seq_veh_in_ped[:-1]
                    
                if args.method == 4:
                    grid_TTC_seq = grid_TTC_seq[:-1]
                    grid_TTC_veh_seq = grid_TTC_veh_seq[:-1]

                #number of peds in this sequence per frame
                numNodes = len(lookup_seq) 
                if lookup_seq_veh is not None:
                    numx_seq_veh = len(lookup_seq_veh)


                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    cell_states = cell_states.cuda()

                # Forward prop
                if args.teacher_forcing:

                    if args.method == 3: # Vanillar LSTM
                        outputs, _, _ = net(x_seq, hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq ,dataloader, lookup_seq) 
                    elif args.method == 4: # Collision Grid
                        outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq ,dataloader,
                                                lookup_seq, x_seq_veh, grid_seq_veh_in_ped, VehsList_seq[:-1], lookup_seq_veh, grid_TTC_seq,
                                                grid_TTC_veh_seq) 
                    elif args.method == 1: # Social LSTM
                        outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq ,dataloader, 
                                                lookup_seq, x_seq_veh, grid_seq_veh_in_ped, VehsList_seq[:-1], lookup_seq_veh) 
                    else:
                        raise ValueError("Method is not defined")
                    
                else: # not teacher forcing

                    outputs = Variable(torch.zeros((args.seq_length-1), numNodes, args.output_size))
                    ret_x_seq = Variable(torch.zeros((args.seq_length), numNodes, x_seq.shape[2]))
                    if args.use_cuda:
                        outputs = outputs.cuda()
                        ret_x_seq = ret_x_seq.cuda()

                    for tstep in range(args.obs_length-1):
                        if args.method == 3: # Vanillar LSTM             
                            output_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numNodes, x_seq.shape[2]),
                                                                            hidden_states, cell_states,
                                                                            [PedsList_seq[tstep]], [numPedsList_seq[tstep]],
                                                                            dataloader, lookup_seq)
                            
                        elif args.method == 4: # Collision Grid
                            output_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numNodes, x_seq.shape[2]),
                                                                            [grid_seq[tstep]], hidden_states, cell_states,
                                                                            [PedsList_seq[tstep]], [numPedsList_seq[tstep]],
                                                                            dataloader, lookup_seq,
                                                                            x_seq_veh[tstep].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                                            [grid_seq_veh_in_ped[tstep]], [VehsList_seq[tstep]], lookup_seq_veh,
                                                                            [grid_TTC_seq[tstep]], [grid_TTC_veh_seq[tstep]])
                        elif args.method == 1: # Social LSTM
                            output_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numNodes, x_seq.shape[2]),
                                                                            [grid_seq[tstep]], hidden_states, cell_states,
                                                                            [PedsList_seq[tstep]], [numPedsList_seq[tstep]],
                                                                            dataloader, lookup_seq,
                                                                            x_seq_veh[tstep].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                                            None, [VehsList_seq[tstep]], lookup_seq_veh)
                        outputs[tstep] = output_obs
                        ret_x_seq[tstep+1,:,:2] = output_obs[:,:,:2] # these are the mean of the gaussian distribution
                    
                    # Last seen grid
                    if args.method != 3: # not vanilla lstm
                        prev_grid = [grid_seq[args.obs_length-1].clone()]
                        if (args.method == 4):
                            prev_grid_veh_in_ped = [grid_seq_veh_in_ped[args.obs_length-1].clone()]
                            prev_TTC_grid = [grid_TTC_seq[args.obs_length-1].clone()]
                            prev_TTC_grid_veh = [grid_TTC_veh_seq[args.obs_length-1].clone()]
                    # last observed position
                    ret_x_seq[args.obs_length-1,:,2:] = x_seq[args.obs_length-1,:,2:] # these are the mean of the gaussian distribution

                    last_observed_frame_prediction = ret_x_seq[args.obs_length-1, :, :2].clone()
                    ret_x_seq[args.obs_length-1, :, :2] = x_seq[args.obs_length-1,:,:2] # storing the last GT observed frame here to ensure this is used in the next for loop and then 
                    # storing the actual prediction in it after the forward network is run for the first step in the prediction length 
                  
                    # rely on the output itself from now on
                    for tstep in range(args.obs_length-1, args.seq_length-1):

                        # froward prop
                        if args.method == 3:
                            outputs_pred, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numNodes, ret_x_seq.shape[2]),
                                                                            hidden_states, cell_states,
                                                                            [PedsList_seq[tstep]], [numPedsList_seq[tstep]],
                                                                            dataloader, lookup_seq) 
                        elif args.method == 4: 
                            outputs_pred, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numNodes, ret_x_seq.shape[2]),
                                                                            prev_grid, hidden_states, cell_states,
                                                                            [PedsList_seq[tstep]], [numPedsList_seq[tstep]],
                                                                            dataloader, lookup_seq,
                                                                            x_seq_veh[tstep].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                                            prev_grid_veh_in_ped, [VehsList_seq[tstep]], lookup_seq_veh,
                                                                            prev_TTC_grid, prev_TTC_grid_veh)
                        elif args.method == 1:
                            outputs_pred, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numNodes, ret_x_seq.shape[2]),
                                                                            prev_grid, hidden_states, cell_states,
                                                                            [PedsList_seq[tstep]], [numPedsList_seq[tstep]],
                                                                            dataloader, lookup_seq,
                                                                            x_seq_veh[tstep].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                                            None, [VehsList_seq[tstep]], lookup_seq_veh)
                        if tstep == args.obs_length-1: 
                            # storing the actual prediction in the last observed frame position
                            ret_x_seq[args.obs_length-1, :, :2] = last_observed_frame_prediction.clone()
                            
                        outputs[tstep] = outputs_pred
                        ret_x_seq[tstep+1,:,:2] = outputs_pred[:,:,:2] # note: the first dimension of ret_x_seq is one more than the outputs

                        # updating the velocity and other features based on the prediction output
                        # order of featurs in x_seq: [x, y, vx, vy, timestamp, ax, ay, speed_change, heading_change, cov11, cov12, cov21, cov22]
                        ret_x_seq_convert = ret_x_seq.clone() 
                        ret_x_seq_convert = revert_postion_change_seq2(ret_x_seq.cpu(), PedsList_seq, lookup_seq,
                                                                        first_values_dict, x_seq_orig, args.obs_length, infer=True)
                        ret_x_seq_convert[tstep+1, :, 2] = (ret_x_seq_convert[tstep+1, :, 0] - ret_x_seq_convert[tstep, :, 0]) / dataloader.timestamp # vx 
                        ret_x_seq_convert[tstep+1, :, 3] = (ret_x_seq_convert[tstep+1, :, 1] - ret_x_seq_convert[tstep, :, 1]) / dataloader.timestamp # vy
                        # updating the velocity data in ret_x_seq accordingly
                        ret_x_seq[tstep+1, :, 2] = ret_x_seq_convert[tstep+1, :, 2].clone()
                        ret_x_seq[tstep+1, :, 3] = ret_x_seq_convert[tstep+1, :, 3].clone()

                        # Extract the mean, std and corr of the bivariate Gaussian
                        mux, muy, sx, sy, corr = getCoef(outputs[tstep].cpu().view(1, numNodes, args.output_size)) # parameters of the gaussian distribution (scaled)
                        scaled_param_dist = torch.stack((mux, muy, sx, sy, corr),2) 
                        cov = cov_mat_generation(scaled_param_dist)
                        ret_x_seq[tstep+1, :, 9:13] = cov.reshape(cov.shape[0], cov.shape[1], 4).squeeze(0) # covariances of the trajectories generated by the predictor

                        if args.method == 1: #social lstm 
                          prev_grid = getSequenceGridMask(ret_x_seq_convert[tstep+1].cpu().view(1, numNodes, ret_x_seq.shape[2]),
                                                            [PedsList_seq[tstep+1]], args.neighborhood_size, args.grid_size,
                                                            args.use_cuda, lookup_seq)
                          
                        elif args.method == 4: #collision grid
                            prev_grid, prev_TTC_grid = getSequenceInteractionGridMask(
                                                                                    ret_x_seq_convert[tstep+1].cpu().view(1, numNodes, ret_x_seq.shape[2]),
                                                                                    [PedsList_seq[tstep+1]], 
                                                                                    ret_x_seq_convert[tstep+1].cpu().view(1, numNodes, ret_x_seq.shape[2]),
                                                                                    [PedsList_seq[tstep+1]], args.TTC,
                                                                                    args.D_min, args.num_sector, args.use_cuda, 
                                                                                    lookup_seq, lookup_seq)
                            prev_grid_veh_in_ped, prev_TTC_grid_veh = getSequenceInteractionGridMask(
                                                                                    ret_x_seq_convert[tstep+1].cpu().view(1, numNodes, ret_x_seq.shape[2]),
                                                                                    [PedsList_seq[tstep+1]],
                                                                                    x_seq_veh[tstep+1].cpu().view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                                                    [VehsList_seq[tstep+1]],
                                                                                    args.TTC_veh, args.D_min_veh, args.num_sector,
                                                                                    args.use_cuda, lookup_seq, lookup_seq_veh,
                                                                                    is_heterogeneous=True, is_occupancy=False) 

                # Compute loss
                loss, NLL_loss, uncertainty_loss = combination_loss_Point2Dist(outputs, y_seq,  PedsList_seq[1:], lookup_seq, mask[1:],
                                                                                args.use_cuda, args.uncertainty_aware)
                # loss, NLL_loss, uncertainty_loss = combination_loss_Dist2Dist(outputs, y_seq, y_dis_cov, PedsList_seq[1:], lookup_seq,
                #                                                                mask[1:], args.use_cuda, args.uncertainty_aware)
                loss = loss / dataloader.batch_size
                loss_batch += loss.item()

                NLL_loss = NLL_loss / dataloader.batch_size
                uncertainty_loss = uncertainty_loss / dataloader.batch_size
                NLL_loss_batch += NLL_loss.item()
                uncertainty_loss_batch += uncertainty_loss.item()

                # Compute gradients
                # Cumulating gradient until we reach our required batch size and then updating one the weights
                loss.backward()
            
                # # Clip gradients 
                # torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                err, pred_seq = sequence_error(outputs.cpu(), x_seq_orig[1:,:,:2], PedsList_seq[1:], lookup_seq, args.use_cuda,
                                                first_values_dict, args.obs_length)
                err_batch += err.item()


            # Update parameters
            optimizer.step()
            
            end = time.time()
            loss_batch = loss_batch 
            err_batch = err_batch / dataloader.batch_size
            err_batch_list.append(err_batch)
            loss_batch_list.append(loss_batch)
            NLL_loss_batch_list.append(NLL_loss_batch)
            uncertainty_loss_batch_list.append(uncertainty_loss_batch)
            loss_epoch += loss_batch
            NLL_loss_epoch += NLL_loss_batch
            uncertainty_loss_epoch += uncertainty_loss_batch
            err_epoch += err_batch
            num_batch+=1

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

            train_batch_num = epoch * dataloader.num_batches + batch
            train_batch_num_list.append(train_batch_num)
            if (train_batch_num%50 == 0):
                Loss_Plot(train_batch_num_list, err_batch_list, loss_batch_list, "loss_plot_batch", "training batch number",
                          NLL_loss_batch_list, uncertainty_loss_batch_list)

        loss_epoch /= dataloader.num_batches
        NLL_loss_epoch /= dataloader.num_batches
        uncertainty_loss_epoch /= dataloader.num_batches
        err_epoch /= dataloader.num_batches
        loss_epoch_list.append(loss_epoch)
        NLL_loss_epoch_list.append(NLL_loss_epoch)
        uncertainty_loss_epoch_list.append(uncertainty_loss_epoch)
        err_epoch_list.append(err_epoch)
        Loss_Plot(range(epoch+1), err_epoch_list, loss_epoch_list, "loss_plot_epoch", "epoch",
                  NLL_loss_epoch_list, uncertainty_loss_epoch_list)

        # Log loss values
        log_file_curve.write("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch)+" error: "+str(err_epoch)+'\n')

        # Save the model after each epoch, with a file name that has the number of epoch at the end of the name (x)
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    end_train_loop = time.time()
    train_time = end_train_loop - start_train_loop
    print("The whole trainig time for {} iteraction was {} seconds".format(args.num_epochs,train_time))


if __name__ == '__main__':
    main()