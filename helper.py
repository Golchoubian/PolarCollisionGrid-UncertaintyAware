
import torch
import numpy as np
from model_CollisionGrid import CollisionGridModel
from model_SocialLSTM import SocialModel
from model_VanillaLSTM import VLSTMModel
from torch.autograd import Variable
import math
import itertools
from scipy.stats import chi2
from matplotlib import pyplot as plt


# one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

def position_change_seq(x_seq, PedsList_seq, lookup_seq):
    #substract each frame value from its previosu frame to create displacment data.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    first_presence_flag = [0]*(x_seq.shape[1])
    latest_pos = [0]*(x_seq.shape[1])

    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            if first_presence_flag[lookup_seq[ped]] == 0: # this frame is the first frame where this pedestrian apears
                vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2] # this should always give (0,0)
                latest_pos[lookup_seq[ped]] = frame[lookup_seq[ped], 0:2]
                first_presence_flag[lookup_seq[ped]] = 1
            else:
                vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - latest_pos[lookup_seq[ped]]
                latest_pos[lookup_seq[ped]] = frame[lookup_seq[ped], 0:2]
    
    return vectorized_x_seq, first_values_dict

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    #substract first frame value to all frames for a ped.Therefore, convert absolute pos. to relative pos.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]
    
    return vectorized_x_seq, first_values_dict


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up, test=False):
    '''
    params:
    outputs : predicted locations
    targets : true locations
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index

    '''
    seq_length = outputs.size()[0]
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    # result = torch.tanh(result) # to ensure that the probability density function passed to log is between 0 and 1., so NLL is always positive

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon)) #, max=1)) # clipping the probability density function to 1 to prevent negative loss
    results = []

    loss = 0
    counter = 0

    for framenum in range(seq_length):

        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]

        for nodeID in nodeIDs:
            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

            if test:
                # also store the individual NLL (results)
                results.append(result[framenum, nodeID].item())
    
    if test:
        return results, loss/counter

    if counter != 0:
        return loss / counter
    else:
        return loss


def get_model(index, arguments, infer = False): 
    # return a model given index and arguments
    if index == 1:
        return SocialModel(arguments, infer)
    elif index == 4:
        return CollisionGridModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    else:
        raise ValueError('Invalid model index!')
   

def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], 
                [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y 

def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]

    return absolute_x_seq



def revert_postion_change_seq2(x_seq, PedsList_seq, lookup_seq, first_values_dict, orig_x_seq, obs_length, infer=False, KF=False):
    # convert displacement array to absolute position array
    absolute_x_seq = x_seq.clone()
    first_presence_flag = [0]*(x_seq.shape[1])
    latest_pos = [0]*(x_seq.shape[1])
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            if first_presence_flag[lookup_seq[ped]] == 0: # this frame is the first frame where this pedestrian apears
                absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]
                latest_pos[lookup_seq[ped]] = absolute_x_seq[ind, lookup_seq[ped], 0:2] 
                # for the first frame this absolute_x_seq is same as the orig_x_seq since frame is [0,0]
                first_presence_flag[lookup_seq[ped]] = 1
            else:
                absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + latest_pos[lookup_seq[ped]]
                if (infer==True and ind>=obs_length) or KF==True: # we have to rely on the algorithm's own prediction for the next state ! thid should be ind>= obs_length
                    latest_pos[lookup_seq[ped]] = absolute_x_seq[ind, lookup_seq[ped], 0:2]
                else: # we use the ground truth that we have
                    latest_pos[lookup_seq[ped]] = orig_x_seq[ind, lookup_seq[ped], 0:2]

    return absolute_x_seq


def get_num_collisions_hetero(ret_nodes, NodesPresent, ObsNodesPresent, using_cuda, look_up, Veh_nodes, VehList, lookup_veh, threshold):


    collision_count = 0
    all_two_cases = 0
    num_nodes = ret_nodes.shape[1]
    num_vehs = Veh_nodes.shape[1]
    two_agent_combination =  itertools.product(list(range(num_nodes)),list(range(num_vehs)))

    lookup_indxToid = reverse_dict(look_up)
    lookup_veh_indxToid = reverse_dict(lookup_veh)

    for ped_ind, veh_ind in two_agent_combination:
        # check if the two agent's have any collision along their predicted trajectory

        idped = lookup_indxToid[ped_ind]
        idveh = lookup_veh_indxToid[veh_ind]

        if (idped not in ObsNodesPresent): 
            # we do not count on the prediction of pedestrians that we did not have any information from them during the observation period
            continue
        
        all_two_cases += 1
        for t in range(len(ret_nodes)):
            if ((idped in NodesPresent[t]) and (idveh in VehList[t])): # if both agent are present in this timestep
                pre_pos_ped = ret_nodes[t][ped_ind][:2]
                pos_veh = Veh_nodes[t][veh_ind][:2]
                dis = pre_pos_ped - pos_veh
                if torch.norm(dis, p=2) < threshold:
                    collision_count += 1
                    break # one time of collision is enough

    return collision_count, all_two_cases



def get_num_collisions_homo(ret_nodes, NodesPresent, ObsNodesPresent, using_cuda, look_up, threshold):


    collision_count = 0
    all_two_cases = 0
    num_nodes = ret_nodes.shape[1]
    two_agent_combination = list(itertools.combinations(list(range(num_nodes)), 2))

    lookup_indxToid = reverse_dict(look_up)

    for two_agent in two_agent_combination:

        # check if the two agent's have any collision along their predicted trajectory
        agentA = two_agent[0]
        agentB = two_agent[1]
        idA = lookup_indxToid[agentA]
        idB = lookup_indxToid[agentB]

        if (idA not in ObsNodesPresent) or (idB not in ObsNodesPresent): 
            # we do not count on the prediction of pedestrians that we did not have any information from them during the observation period
            continue
        
        all_two_cases += 1
        for t in range(len(ret_nodes)):
            if ((idA in NodesPresent[t]) and (idB in NodesPresent[t])): # if both agent are present in this timestep
                pre_posA = ret_nodes[t][agentA][:2]
                pre_posB = ret_nodes[t][agentB][:2]
                dis = pre_posB - pre_posA
                if torch.norm(dis, p=2) < threshold:
                    collision_count += 1
                    break # one time of collision is enough

    return collision_count, all_two_cases


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()

    for tstep in range(pred_length):
        counter = 0

        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)

           
            if nodeID not in ObsNodesPresent: 
                # This is for elimiating those agents that did not have any observation data 
                # and only appread during the predictino length. We want to exclude then from the error calculation process
                continue

            nodeID = look_up[nodeID]


            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)



def get_final_error(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index


    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = int(nodeID)


        if nodeID not in ObsNodesPresent: # When this will happen?!
            continue

        nodeID = look_up[nodeID]

        
        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        
        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1
        
    if counter != 0:
        error = error / counter
            
    return error


def sequence_error(outputs, x_orig, Pedlist, look_up, using_cuda, first_values_dict,obs_length):

    seq_len = outputs.shape[0]
    num_ped = outputs.shape[1]
    pred_seq = Variable(torch.zeros(seq_len, num_ped, 2))
    if using_cuda:
        pred_seq = pred_seq.cuda()

    for tstep in range(seq_len):
        mux, muy, sx, sy, corr = getCoef(outputs[tstep,:,:].view(1, num_ped, 5))
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
        pred_seq[tstep,:,0] = next_x
        pred_seq[tstep,:,1] = next_y

    Pedlist_whole_seq = sum(Pedlist, [])
    
    # When working with displacement the following two lines should be added:
    pred_seq_abs = revert_postion_change_seq2(pred_seq.data.cpu(), Pedlist, look_up, first_values_dict, x_orig, obs_length)
    total_error = get_mean_error(pred_seq_abs.data, x_orig.data, Pedlist, Pedlist_whole_seq, using_cuda, look_up)

    return total_error, pred_seq_abs

def get_hausdorff_distance(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : The largest distance from each predicted point to any point on the ground truth trajectory (Modified Hausdorff Distance: MHD)
    '''
    pred_length = ret_nodes.size()[0]
    PedsPresent = list(set(sum(assumedNodesPresent, []))) # getting a list of all non repeating agetn ids present in the prediction length
    Valid_PedsPresent = [i for i in PedsPresent if i in ObsNodesPresent] # elimiating those agents that did not have any observation data and
    # only appread during the prediction length
    num_agents = len(Valid_PedsPresent)

    if (num_agents == 0):
        return None
    else:
        error = torch.zeros(num_agents)
    
        if using_cuda:
            error = error.cuda()

        count = 0
        for nodeID in Valid_PedsPresent:
        
            nodeID = int(nodeID)
            present_frames = [i for i, id_list in enumerate(assumedNodesPresent) if nodeID in id_list]
            
            nodeID = look_up[nodeID]
            pred_traj = ret_nodes[present_frames, nodeID, :]
            true_traj = nodes[present_frames, nodeID, :]

            error_t = torch.zeros(len(present_frames))
            for tstep in range(len(present_frames)):
                pred_pos = pred_traj[tstep,:]
                error_t[tstep] = torch.max(torch.norm(true_traj - pred_pos, p=2, dim=1)) 
                # the maximum distance between this predicted timestep and the ground truth trajectory
            
            # maximum among all the predicted time steps distance to groud truth for this agent
            error[count] = max(error_t)
            count += 1

        MHD = sum(error)/count

        return MHD



def get_velocity_errors(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean distance between predicted speed and true speed & Mean distance between predicted heaing and true heading
    '''
    pred_length = ret_nodes.size()[0]
    error_speed = torch.zeros(pred_length)
    error_heading = torch.zeros(pred_length)

    if using_cuda:
        error_speed = error_speed.cuda()
        error_heading = error_heading.cuda()

    for tstep in range(pred_length): 
        counter = 0

        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)

          
            if nodeID not in ObsNodesPresent:
                continue

            nodeID = look_up[nodeID]


            pred_v = ret_nodes[tstep, nodeID, :]
            true_v = nodes[tstep, nodeID, :]

            pred_speed = torch.norm(pred_v, p=2)
            true_speed = torch.norm(true_v, p=2)  

            error_speed[tstep] += torch.pow((pred_speed - true_speed), 2)

            dot = true_v[0]*pred_v[0] + true_v[1]*pred_v[1] # dot is propotional to cos(theta)
            det = true_v[0]*pred_v[1] - pred_v[0]*true_v[1] # det (|V_e V_n|) is propotional to sin(theta)
            angle = math.atan2(det,dot) * (180/math.pi) # the value is between -180 and 180
            error_heading[tstep] += angle**2

            counter += 1

        if counter != 0:
            error_speed[tstep] = error_speed[tstep] / counter
            error_heading[tstep] = error_heading[tstep] / counter

    return torch.mean(error_speed), sum(error_heading)/len(error_heading)

def get_mean_NLL(dist_param, orig_x, PedsList, PedsList_obs, using_cuda, lookup):
    '''
    The output is a list of individual NLL for the exsting peds at each timestep.
    being stored to later calculate the mean and std of the whole trajeoctries in the test set
    '''

    # exclud those peds from the PedList that we did not have any observation data from them
    for f in range(len(PedsList)):
        PedsList[f] = [i for i in PedsList[f] if i in PedsList_obs]
    
    NLL_list, NLL_loss = Gaussian2DLikelihood(dist_param, orig_x, PedsList, lookup, test=True) # the output being a list of NLL
    return NLL_list, NLL_loss.item()



def within_sigma_levels(point, mean, covariance):

    '''
    Checking if a given point (GT here) falls into
    i sigma level of the predicted distribution given its mean and covaraince
    where i is 1, 2, and 3
    The output is a boolean vector of size 3 
    '''

    # Calculate the Mahalanobis distance
    # delta = point - mean
    # inv_covariance = torch.inverse(covariance)
    # mahalanobis_distance = torch.sqrt(torch.matmul(torch.matmul(delta, inv_covariance), delta))

    _mahalanobis_distance = mahalanobis_distance(point, mean, covariance).item()

    # Determine the critical values for 1-sigma, 2-sigma, and 3-sigma levels
    dim = len(mean)
    critical_value_1sigma = np.sqrt(chi2.ppf(0.68, dim))
    critical_value_2sigma = np.sqrt(chi2.ppf(0.95, dim))
    critical_value_3sigma = np.sqrt(chi2.ppf(0.997, dim))

    # chi2_distribution =  torch.distributions.Chi2(df=dim)
    # critical_value_1sigma = torch.sqrt(chi2_distribution.icdf(torch.tensor(0.68)))
    # critical_value_2sigma = torch.sqrt(chi2_distribution.icdf(torch.tensor(0.95)))
    # critical_value_3sigma = torch.sqrt(chi2_distribution.icdf(torch.tensor(0.997)))

    # Check if the Mahalanobis distance is within the sigma levels
    within_1sigma = int(_mahalanobis_distance <= critical_value_1sigma)
    within_2sigma = int(_mahalanobis_distance <= critical_value_2sigma)
    within_3sigma = int(_mahalanobis_distance <= critical_value_3sigma)

    return within_1sigma, within_2sigma, within_3sigma



def Delta_Empirical_Sigma_Value(dist_param, orig_x, PedsList, PedsList_obs, using_cuda, lookup):
    '''
    The difference in the fraction of GT positions that fall within the
    i-sigma level set of the predicted distribution and the fraction from
    an ideal Gaussian
    '''
    mux, muy, sx, sy, corr = getCoef(dist_param)
    _dist_param = torch.stack((mux, muy, sx, sy, corr),2) 

    cov_matrix = cov_mat_generation(_dist_param)
    # exclud those peds from the PedList that we did not have any observation data from them
    for f in range(len(PedsList)):
        PedsList[f] = [i for i in PedsList[f] if i in PedsList_obs]

    counter = 0
    is_in_1_sigma = 0
    is_in_2_sigma = 0
    is_in_3_sigma = 0
    
    for f in range(len(PedsList)):
        for ped in PedsList[f]:
            counter += 1 
            ped_indx = lookup[ped]
            is_in_1_sigma += within_sigma_levels(orig_x[f,ped_indx,:],
                                                _dist_param[f, ped_indx, 0:2], cov_matrix[f,ped_indx])[0]
            is_in_2_sigma += within_sigma_levels(orig_x[f,ped_indx,:],
                                                _dist_param[f, ped_indx, 0:2], cov_matrix[f,ped_indx])[1]
            is_in_3_sigma += within_sigma_levels(orig_x[f,ped_indx,:], 
                                                _dist_param[f, ped_indx, 0:2], cov_matrix[f,ped_indx])[2]
            
    # leave the calculation of the fraction to when we have the predictions for all the data in test set
    # and just report the difference and the counter here
    return is_in_1_sigma, is_in_2_sigma, is_in_3_sigma, counter

    # fraction_1sigma = is_in_1_sigma/counter
    # fraction_2sigma = is_in_2_sigma/counter
    # fraction_3sigma = is_in_3_sigma/counter

    # diff_1sigma = fraction_1sigma - 0.68
    # diff_2sigma = fraction_2sigma - 0.95
    # diff_3sigma = fraction_3sigma - 0.997

    # return diff_1sigma, diff_2sigma, diff_3sigma



def available_frame_extraction(agent_indx, pedlist_seq, lookup_seq):
    
    key_list = list(lookup_seq.keys())
    val_list = list(lookup_seq.values())
    indx_position = val_list.index(agent_indx)
    agent_id = key_list[indx_position]

    present_frames = []

    for f in range(len(pedlist_seq)):
        if agent_id in pedlist_seq[f]:
            present_frames.append(f)

    return present_frames



def reverse_dict(lookup):

    reversedDict = dict()
    key_list = list(lookup.keys()) # the agent id
    val_list = list(lookup.values()) # the index number of that agent in the tensor
    n = len(key_list)
    for i in range(n):
        key = val_list[i]
        val = key_list[i]
        reversedDict[key] = val
    
    return reversedDict

def cov_mat_generation(dist_param): # the input should be dist_param after the scaling has been done in the gen_Coef
        
    '''
    Generating the covanriance matrix from the distribution parameters
    dist_param: numpy array of shape: (pred_seq_len, num_peds, 5)
    bi-varainat gaussian distribution parameters in the third dimesnion are ordered as follows:
    [mu_x, mu_y, sigma_x, sigma_y, rho]
    Output: numpy array of shape: (pred_seq_len, num_peds, 2, 2)
    '''
    
    mu_x = dist_param[:, :, 0]
    mu_y = dist_param[:, :, 1]
    sigma_x = dist_param[:, :, 2]
    sigma_y = dist_param[:, :, 3]
    rho = dist_param[:, :, 4]

    # compute the element of the covariance matrix
    sigma_x2 = sigma_x ** 2
    sigma_y2 = sigma_y ** 2
    sigma_xy = sigma_x * sigma_y
    rho_sigma_xy = rho * sigma_xy

    # create the convanriance matrix tensor
    cov_mat = torch.stack([
        torch.stack([sigma_x2, rho_sigma_xy], dim=-1),
        torch.stack([rho_sigma_xy, sigma_y2], dim=-1)
        ], dim=-2)

    # test_cov = torch.zeros((cov_mat.shape[0], cov_mat.shape[1], 2, 2))
    # for t in range(cov_mat.shape[0]):
    #     for ped in range(cov_mat.shape[1]):
    #         test_cov[t,ped,:,:] = torch.tensor([[sigma_x2[t, ped], rho_sigma_xy[t, ped]], [rho_sigma_xy[t, ped], sigma_y2[t, ped]]])
    
    
    # print('==== check rho ====')
    # print('rho:', rho[-1,-3])
    # # print('>>>>> compare <<<<<')
    # # print('cov_mat:', cov_mat[-1,-3,:,:])
    # # print('test_cov:', test_cov[-1,-3,:,:])
    
    # I guess there is no need for worrying about the non-existing peds of the corrent steps in the output
    # as the distribution parameters for those peds are all zeros and the cov_mat will be all zeros as well
    # each row is the cov_mat of the prediction we have for that specifc time step
    
    return cov_mat


def mahalanobis_distance(GT, mean, covariance):
    '''
    Calculate the Mahalanobis distance
    params:
    GT: ground truth locations of shape (num_frames, num_pedestrian, 2)
    mean: mean of the predicted locations of shape (num_frames, num_pedestrian, 2)
    covariance: covariance matrix of the predicted locations of shape (num_frames, num_pedestrian, 2, 2)
    output: mahalanobis distance of shape (num_frames, num_pedestrian)
    '''
    delta =  mean - GT
    
    # Reshape the difference tensor to (num_frames, num_pedestrian, 2, 1)
    delta = delta.unsqueeze(-1)

    # Calculate the inverse of the covariance matrix
    covariance_inverse = torch.inverse(covariance)

    # Compute the Mahalanobis distance
    mahalanobis_dist = torch.matmul(covariance_inverse, delta)
    mahalanobis_dist = torch.matmul(delta.transpose(-1, -2), mahalanobis_dist).squeeze(-1) # transpose should be between -2,-3
    mahalanobis_dist = torch.sqrt(mahalanobis_dist).squeeze(-1)


    return mahalanobis_dist

# def probability_within_confidence_interval(mahalanobis_dist, dof=2):
#     # Calculate the probability using the chi-squared CDF
#     probability = chi2.cdf(mahalanobis_dist, df=dof)

#     return probability


def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """
    Calculate the Bhattacharyya distance between two bivariate Gaussian distributions.

    Parameters:
    - mu1: Mean of the first distribution (Tensor of shape (seq_lenght, num_peds, 2)).
    - cov1: Covariance matrix of the first distribution (Tensor of (seq_lenght, num_peds, 2, 2)).
    - mu2: Mean of the second distribution (Tensor of shape (seq_lenght, num_peds, 2)).
    - cov2: Covariance matrix of the second distribution (Tensor of (seq_lenght, num_peds, 2, 2)).

    Returns:
    - Bhattacharyya distance: A tensor of shape [seq_length, num_peds]
    """

    # Calculate Bhattacharyya coefficient
    cov_avg = 0.5 * (cov1 + cov2)
    diff = mu2 - mu1

    diff = diff.unsqueeze(-1)

    mahalanobis_sq = torch.matmul(diff.transpose(-1,-2), torch.matmul(torch.inverse(cov_avg), diff)).squeeze(-1).squeeze(-1)  
    bhattacharyya_coeff = 0.125 * mahalanobis_sq + 0.5 * torch.logdet(cov_avg) - 0.25 * (torch.logdet(cov1) + torch.logdet(cov2))

    # print('cov2:', cov2.reshape(cov2.shape[0], cov2.shape[1],4))
    # print('det(cov2):', torch.det(cov2))

    # Calculate Bhattacharyya distance
    bhatt_distance = bhattacharyya_coeff # -torch.log(torch.exp(-bhattacharyya_coeff))

    # print('bhatt_distance:', bhatt_distance)

    return bhatt_distance

def uncertainty_aware_loss_Point2Dist(outputs, targets, mask, use_cuda):

    '''
    params:
    outputs : bi-variant Gaussian distribution paramters of the predicted locations
    targets : true locations
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index

    '''
    seq_length = outputs.size()[0]
    num_peds = outputs.size()[1]

    degrees_of_freedom = torch.tensor(2, dtype=torch.int32) # for 2D bi-variant Gaussian distribution

    mux, muy, sx, sy, corr = getCoef(outputs)
    dist_param_seq = torch.stack((mux, muy, sx, sy, corr),2) 
    cov_matrix = cov_mat_generation(dist_param_seq)
    predicted_covs = torch.tile(torch.tensor([[0.01, 0.0], [0.0, 0.01]]), (seq_length, num_peds, 1, 1))
    if use_cuda:
        predicted_covs = predicted_covs.cuda()
        degrees_of_freedom = degrees_of_freedom.to('cuda')

    for ped_indx in range(num_peds):    
        # find the frames that this ped is present using nodesPresent
        present_frames = torch.nonzero(mask[:, ped_indx]).squeeze(1)
        predicted_covs[present_frames, ped_indx, :] = cov_matrix[present_frames, ped_indx,:] # for those peds that have no data, this keeps the default value of 0.01 !!!!!

    mahalanobis_dist = mahalanobis_distance(targets, outputs[:,:,:2], predicted_covs) # this should be tensor of same size as mask

    
    # # Calculate the probability density function (PDF) for the given Mahalanobis distance (a chi2-squared distribution)
    # # we want to minimize the mahalanobis distance, so we want to maximize the pdf_value
    # # Create a Chi2 distribution with the specified degrees of freedom
    # chi2_distribution = torch.distributions.Chi2(degrees_of_freedom)
    # # Calculate the PDF for each element of the tensor
    # pdf_values = chi2_distribution.log_prob(mahalanobis_dist).exp()
    
    # # Numerical stability
    # epsilon = 1e-20
    # result = -torch.log(torch.clamp(pdf_values, min=epsilon))

    result = mahalanobis_dist # testing minimizing the mahalanobis distance itself instead of its -log(probability)


    result = result * mask # this will make those elements that do not existm don't count (those peds that have no data)
    # sum all the numbers in the tensor
    loss = torch.sum(result)
    # the overall distances for all the peds in the sequence
    counter = torch.sum(mask)
    if counter != 0:
        return loss / counter
    else:
        return loss # if loss is a valid number! otherwise we have to return 0
    

def uncertainty_aware_loss_Dist2Dist(outputs, targets_mean, targets_cov, mask, use_cuda):

    '''
    Calculating the mahalanobis distance between the ground truth distribution and the predicted distribution
    params:
    outputs : bi-variant Gaussian distribution paramters of the predicted locations
    targets : true locations (mean and distribution)
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index
    '''
    seq_length = outputs.size()[0]
    num_peds = outputs.size()[1]

    mux, muy, sx, sy, corr = getCoef(outputs)
    dist_param_seq = torch.stack((mux, muy, sx, sy, corr),2) 
    cov_matrix = cov_mat_generation(dist_param_seq)
    predicted_covs = torch.tile(torch.tensor([[0.01, 0.0], [0.0, 0.01]]), (seq_length, num_peds, 1, 1))
    if use_cuda:
        predicted_covs = predicted_covs.cuda()

    for ped_indx in range(num_peds):    
        # find the frames that this ped is present using nodesPresent
        present_frames = torch.nonzero(mask[:, ped_indx]).squeeze(1)
        predicted_covs[present_frames, ped_indx, :] = cov_matrix[present_frames, ped_indx,:] # for those peds that have no data, this keeps the default value of 0.01 !!!!!

    Bh_dist = bhattacharyya_distance(targets_mean, targets_cov, outputs[:,:,:2], predicted_covs) # this should be tensor of same size as mask
    # Bh_coef = torch.exp(-Bh_dist/4) # the value is between 0 and 1 (the maximum value is 1 when Bh_dist is 0)

    # # Numerical stability
    # epsilon = 1e-20
    # result = -torch.log(torch.clamp(Bh_coef, min=epsilon)) # this is the negative log of the Bhattacharyya coefficient
    # # Bh_dist = -torch.log(torch.clamp((1-Bh_dist), min=epsilon)) # considering that Bh_dist remains between 0 and 1 as I see in my simulation

    result = Bh_dist * mask # this will make those elements that do not exist, don't count (those peds that have no data)
    # sum all the numbers in the tensor
    loss = torch.sum(result)
    # the overall distances for all the peds in the sequence
    counter = torch.sum(mask)
    if counter != 0:
        return loss / counter
    else:
        return loss # if loss is a valid number! otherwise we have to return 0
    


def combination_loss_Point2Dist(outputs, targets, nodesPresent, look_up, mask, use_cuda ):

    NLL_loss = Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up)
    uncertainty_loss = uncertainty_aware_loss_Point2Dist(outputs, targets, mask, use_cuda)
    loss = NLL_loss + uncertainty_loss

    return loss, NLL_loss, uncertainty_loss


def combination_loss_Dist2Dist(outputs, targets_mean, targets_cov, nodesPresent, look_up, mask, use_cuda):
   
    w = 10
    NLL_loss = Gaussian2DLikelihood(outputs, targets_mean, nodesPresent, look_up)
    uncertainty_loss = uncertainty_aware_loss_Dist2Dist(outputs, targets_mean, targets_cov, mask, use_cuda)
    loss = NLL_loss + w * uncertainty_loss # might need a weight for each loss in this case since they might not be in the same scale
   
    return loss, NLL_loss, w*uncertainty_loss



def KF_covariance_generator(x_seq, mask, dt, plot_bivariate_gaussian3=None, ax2=None,
                            x_orig=None, Pedlist=None, lookup=None, use_cuda=None,
                            first_values_dict=None, obs_length=None):

    '''
    This function is used to generate a distribution around each
    ground truth data using Kalman Filter.
    It ouptus the filtered states and the covariance matrix of the distribution
    Outputs:
    filtered_states: tensor of shape (seq_length, num_peds, 2)
    filtered_covariances: tensor of shape (seq_length, num_peds, 2, 2)
    '''
    # Note: !!!!!!!!! for now we are not using the mask data, considering that the data of all the peds are available in each frame !!!!!!!!!!!!!!!!!

    # parameters to adjust later
    process_noise_std = 0.5
    measurement_noise_std = 0.7
    
    # this should be done before making the x_seq as position change
    seq_length = x_seq.shape[0]
    num_peds = x_seq.shape[1]

    # Initial state and covariance for each ped
    initial_state = x_seq[0,:,:4]
    initial_covariance = torch.tile(torch.eye(4), (num_peds, 1, 1))

    # Process and measurement noise covariance matrices
    process_noise = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(num_peds, 1, 1) * process_noise_std**2
    measurement_noise = torch.eye(2, dtype=torch.float32).view(1, 2, 2).repeat(num_peds, 1, 1) * measurement_noise_std**2

    # State transition matrix and measurement matrix
    F = torch.tensor([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32).view(1, 4, 4).repeat(num_peds, 1, 1)
    H = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32).view(1, 2, 4).repeat(num_peds, 1, 1)

    measurements = x_seq[1:, :, :2] # only position data is used for measurement

    # Run Kalman filter
    filtered_states, filtered_covariances = kalman_filter(
        initial_state, initial_covariance, measurements, measurement_noise, process_noise, F, H)

    # # ============== test the output by plotting ============
    # filtered_states_abs = revert_postion_change_seq2(filtered_states.data.cpu(), Pedlist, lookup, 
    #                                                  first_values_dict, x_orig, obs_length, KF=True)
    # for agent_index in range(x_seq.shape[1]):
    #     for f in range(x_seq.shape[0]):
    #         if mask[f,agent_index] != 0:
    #             mean = filtered_states_abs[f,agent_index,:2]
    #             cov = filtered_covariances[f,agent_index,:2,:2]
    #             plot_bivariate_gaussian3(mean, cov, ax2, 1)
    #             ax2.plot(mean[0], mean[1], c='b', marker="s", markersize=1)
    #             ax2.plot(x_orig[f,agent_index,0], x_orig[f,agent_index,1], c='r', marker="s", markersize=1)
    #     if agent_index == 0:
    #         ax2.plot(filtered_states_abs[:,agent_index,0], filtered_states_abs[:,agent_index,1], c='b', ls="-", linewidth=1.0, label="filtered")
    #         ax2.plot(x_orig[:,agent_index,0], x_orig[:,agent_index,1], c='r', ls="-", linewidth=1.0, label="ground truth / measurement")
    #         ax2.legend()
    #     else: 
    #         ax2.plot(filtered_states_abs[:,agent_index,0], filtered_states_abs[:,agent_index,1], c='b', ls="-", linewidth=1.0)
    #         ax2.plot(x_orig[:,agent_index,0], x_orig[:,agent_index,1], c='r', ls="-", linewidth=1.0)

    # plt.show()
    # plt.pause(10)
    # plt.cla()
    # # =====================================================
    
    return filtered_states[:,:,:2], filtered_covariances[:,:,:2,:2] # only passing the position covariance matrix




def kalman_filter(x, P, measurements, R, Q, F, H): 
    # !!! Remember to do it only for those exisitng. You might be able to take care of this in the KF_covariance_generator function !!
    """
    Kalman Filter implementation

    Parameters:
    - x: Initial state estimate
    - P: Initial state covariance matrix
    - measurements: List of measurements over time !!!! this is a torch tensor not a list !!!!!!!!!!!!!!!
    - R: Measurement noise covariance matrix
    - Q: Process noise covariance matrix
    - F: State transition matrix
    - H: Measurement matrix

    Returns:
    - filtered_states: tensor of filtered state estimates of dimension (seq_len, num_peds, 4)
    - filtered_covariances: tensor of filtered state covariances of dieemnsion (seq_len, num_peds, 4, 4)
    """
    x = x.unsqueeze(2) 
    filtered_states = [x]
    filtered_covariances = [P]
 

    for z in measurements: # this is going step by step over the sequence length

        z = z.unsqueeze(2)

        # Prediction Step
        x = F @ x
        P = F @ P @ F.transpose(-1, -2) + Q

        # Update Step
        innovation = z - H @ x
        S = H @ P @ H.transpose(-1, -2) + R
        K = P @ H.transpose(-1, -2) @ torch.inverse(S)


        x = x + K @ innovation
        P = (torch.eye(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1)- K @ H) @ P 

        # for i in range(S.shape[0]):
        #     print('=================')
        #     print('P:', P[i,:,:])
        # print('_______________________')

        # print('P:', P[-1,:,:2]) # the covariance matrix of all pedestrians are the same in each time step !! The uncertainty also reduces over time !!

        filtered_states.append(x)
        filtered_covariances.append(P)

    # print('=================')

    # covnert the list of tensors to a single tensor of size (seq_len, num_peds, 2,2) for the convariance
    filtered_covariances = torch.stack(filtered_covariances, dim=0)
    # also convert the list of tesors for the filtered state to a single tensor of size (seq_len, num_peds, 4)
    filtered_states = torch.stack(filtered_states, dim=0)

    return filtered_states.squeeze(-1), filtered_covariances