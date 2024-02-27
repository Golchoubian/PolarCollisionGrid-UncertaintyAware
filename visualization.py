
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import string
from matplotlib.patches import Ellipse
from helper import cov_mat_generation, getCoef
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_trajecotries(true_trajectories,pred_trajectories,obs_len,batch,
                      dist_param_seq,PedsList_seq,lookup,true_trajectories_veh=None,
                      VehsList_seq=None,lookup_seq_veh=None, grid_seq=None, 
                      grid_seq_veh=None, is_train=False, frame_i=5):

    num_ped = pred_trajectories.shape[1]
    seq_len = pred_trajectories.shape[0]
    if true_trajectories_veh is not None:
        num_veh = true_trajectories_veh.shape[1]
    else:
        num_veh = 0
   

    fig, ax = plt.subplots()
    plt.ion()
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    min_x = pred_trajectories[:,:,0].min() - 10
    max_x = pred_trajectories[:,:,0].max() + 10
    min_y = pred_trajectories[:,:,1].min() - 10
    max_y = pred_trajectories[:,:,1].max() + 10

    # # scenario 500
    # min_x = 55
    # max_x = 78
    # min_y = 35
    # max_y = 55

    # # scenario 621
    # min_x = 48
    # max_x = 78
    # min_y = 25
    # max_y = 55

    # # scenario 621 (zoomed on the stationary ped)
    # min_x = 57
    # max_x = 63
    # min_y = 43
    # max_y = 48
    
    # # scenario 730
    # min_x = 42
    # max_x = 62
    # min_y = 25
    # max_y = 45

    # scenario 874
    min_x = 43
    max_x = 66
    min_y = 27 # 26
    max_y = 50 # 49

    # # scenario 871
    # min_x = 42.5
    # max_x = 67.5
    # min_y = 26 # 26
    # max_y = 53 # 49


    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    mux, muy, sx, sy, corr = getCoef(dist_param_seq)
    scaled_param_dist = torch.stack((mux, muy, sx, sy, corr), 2)
    cov_matrix = cov_mat_generation(scaled_param_dist)


    for agent_index in range(num_ped): # for each agent plotting its trajecotry for the frames the agent is present

        # finding the id of the agent usign the lookup dictionary, 
        # this is the opposite of what we were usally doing (geting the index from the agent_id in ped_list_seq)
        id_list = list(lookup.keys())
        index_list = list(lookup.values())
        position = index_list.index(agent_index)
        agent_id = id_list[position]

        # Plotting the observation part
        # frames that the agent is present 
        obs_frame_nums = []
        for frame in range(0,obs_len):
            if agent_id in PedsList_seq[frame]:
                obs_frame_nums.append(frame)

        pred_frame_nums = []
        for frame in range(obs_len,seq_len):
            if agent_id in PedsList_seq[frame]:
                pred_frame_nums.append(frame)

        ### Ground truth:           
        alpha_val = 1.0 # transparancy value
        # observed traj
        ax.plot(true_trajectories[obs_frame_nums,agent_index,0], true_trajectories[obs_frame_nums,agent_index,1],
                  c='g', linewidth=2.0, alpha=alpha_val, ls='--')
        # ground truth predcition
        ax.plot(true_trajectories[pred_frame_nums,agent_index,0], true_trajectories[pred_frame_nums,agent_index,1],
                  c='0.0', linewidth=2.0, alpha=alpha_val, ls='-')
            
        ### Predictions

        max_size = 6
        min_size = 1
        # prediction of CollisionGrid
        for f in pred_frame_nums:
            marker_size = min_size + ((max_size-min_size)/seq_len * f)
            # plot also the current positions 1 sigma confidence interval
            mean = dist_param_seq[f,agent_index,:2]
            cov = cov_matrix[f,agent_index]
            plot_bivariate_gaussian3(mean, cov, ax, 1)
        ax.plot(pred_trajectories[pred_frame_nums,agent_index,0], pred_trajectories[pred_frame_nums,agent_index,1],
                  c='r', linestyle = (0,(1,0.7)), linewidth=2)


        zoomed_ped_id = 3
        if agent_index == zoomed_ped_id: 
            # ============ zoomed in plot ============
            # Define the inset axis position
            ax_inset = inset_axes(ax, width='35%', height='35%', loc='upper left') 

            # Plot the zoomed-in data
            for f in pred_frame_nums:
                marker_size = min_size + ((max_size-min_size)/seq_len * f)
                mean = dist_param_seq[f,agent_index,:2]
                cov = cov_matrix[f,agent_index]
                plot_bivariate_gaussian3(mean, cov, ax_inset, 1)
            ax_inset.plot(pred_trajectories[pred_frame_nums,agent_index,0], 
                        pred_trajectories[pred_frame_nums,agent_index,1],
                        c='r', linestyle = (0,(1,0.7)), linewidth=2.5,
                        label='Zoomed In')

            # observed traj
            ax_inset.plot(true_trajectories[obs_frame_nums,agent_index,0],
                           true_trajectories[obs_frame_nums,agent_index,1],
                            c='g', linewidth=2.0, alpha=alpha_val, ls='--')
            # ground truth predcition
            ax_inset.plot(true_trajectories[pred_frame_nums,agent_index,0], 
                           true_trajectories[pred_frame_nums,agent_index,1],
                            c='0.0', linewidth=2.0, alpha=alpha_val, ls='-')
            # current position
            ax_inset.plot(true_trajectories[frame_i, agent_index,0],
                           true_trajectories[frame_i, agent_index,1], 
                            color='k', marker="*", markersize=6)
            
            # sceanrio 874
            ax_inset.set_xlim(50.5, 51.8)  # Set x-axis range
            ax_inset.set_ylim(37.7, 39)  # Set y-axis range
            
            # # sceanrio 871
            # ax_inset.set_xlim(50.5, 51.5)  # Set x-axis range
            # ax_inset.set_ylim(38, 39)  # Set y-axis range
            # # # sceanrio 871 whole cov
            # # ax_inset.set_xlim(50, 52.5)  # Set x-axis range
            # # ax_inset.set_ylim(36, 40.5)  # Set y-axis range

            # Remove the axes labels in the zoomed-in plot
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_xlabel('')
            ax_inset.set_ylabel('')
            # Mark the area of the inset in the main plot
            mark_inset(ax, ax_inset, loc1=3, loc2=1, fc="none", ec="0.5")

      
    max_size_veh = 6
    min_size_veh = 2

    for veh_ind in range(num_veh):

        id_list_veh = list(lookup_seq_veh.keys())
        index_list_veh = list(lookup_seq_veh.values())
        position_veh = index_list_veh.index(veh_ind)
        veh_id = id_list_veh[position_veh]
        pres_frame_nums = []
        for frame in range(seq_len):
            if veh_id in VehsList_seq[frame]:
                pres_frame_nums.append(frame)
                marker_size = min_size_veh + ((max_size_veh-min_size_veh)/seq_len * frame)
                ax.plot(true_trajectories_veh[frame,veh_ind,0], true_trajectories_veh[frame,veh_ind,1], 
                         c='0.5', marker='o', markersize=marker_size)
        ax.plot(true_trajectories_veh[pres_frame_nums,veh_ind,0], true_trajectories_veh[pres_frame_nums,veh_ind,1], 
                 c='0.5', linewidth=2.0)

    
    # ===================================================================
    # ============================= Neigbors ============================
    # ===================================================================

    alphabet = list(string.ascii_uppercase)
    label_font_size = 12


    if grid_seq != None:

        ego_agent_indx_in_pedlist = 0

        for indx, nodes_pre in enumerate(PedsList_seq[frame_i]):
            agent_i = lookup[nodes_pre]
            label = "Ped " + alphabet[indx]

            # ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
            #             color='k', marker="*", markersize=9)
            # ax.text(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1],
            #     label, fontsize = label_font_size, color ="k") 
            
            if agent_i < 7:
                if indx in [1]:
                    ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
                            color='k', marker="*", markersize=9)
                    ax.text(true_trajectories[frame_i, agent_i,0]+0.5, true_trajectories[frame_i, agent_i,1]+0.7,
                        label, fontsize = label_font_size, color ="k") 
                elif indx in [2]:
                    ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
                                color='k', marker="*", markersize=9)
                    ax.text(true_trajectories[frame_i, agent_i,0]+0.4, true_trajectories[frame_i, agent_i,1]+0.4,
                            label, fontsize = label_font_size, color ="k") 
                elif indx in [3]:
                    ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
                            color='k', marker="*", markersize=9)
                    ax.text(true_trajectories[frame_i, agent_i,0]-0.4, true_trajectories[frame_i, agent_i,1]+1,
                        label, fontsize = label_font_size, color ="k") 
                elif indx in [4]:
                    ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
                            color='k', marker="*", markersize=9)
                    ax.text(true_trajectories[frame_i, agent_i,0]+0.5, true_trajectories[frame_i, agent_i,1],
                        label, fontsize = label_font_size, color ="k") 
                elif indx in [5]:
                    ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
                            color='k', marker="*", markersize=9)
                    ax.text(true_trajectories[frame_i, agent_i,0]-1.4, true_trajectories[frame_i, agent_i,1]+0.5,
                        label, fontsize = label_font_size, color ="k") 
                else:
                    ax.plot(true_trajectories[frame_i, agent_i,0], true_trajectories[frame_i, agent_i,1], 
                                color='k', marker="*", markersize=9)
                    ax.text(true_trajectories[frame_i, agent_i,0]-0.5, true_trajectories[frame_i, agent_i,1]+0.7,
                            label, fontsize = label_font_size, color ="k") 
            


        for indx, nodes_pre in enumerate(VehsList_seq[frame_i]):
            label_veh = "Veh " + alphabet[indx]
            agent_i = lookup_seq_veh[nodes_pre]
            ax.plot(true_trajectories_veh[frame_i, agent_i,0], true_trajectories_veh[frame_i, agent_i,1], 
                         color='0.3', marker="*", markersize=11)
            ax.text(true_trajectories_veh[frame_i, agent_i,0]-0.5, true_trajectories_veh[frame_i, agent_i,1]-1.5,
                        label_veh, fontsize = label_font_size, color ="k") 

    # legends
    ax.plot(-100,-100, c='b', ls='-', label='$1\sigma$ std. pred')
    ax.plot(-100,-100, c='r', ls=':', label='UAW-PCG mean pred.')
    # ax.plot(-100,-100, c='r', ls=':', label='PCG mean pred.')
    ax.plot(-100,-100, c='k', ls='-', label='Ground truth')
    ax.plot(-100,-100, c='g', ls='--', label='Observed traj.')

    ax.legend(loc="lower right", prop={'size': 12}, ncol=2)

    
    if is_train:
        plt.savefig("Store_Results/plot/train/plt/compare/%d.png"%batch, dpi=200)
    else:
        plt.savefig("Store_Results/plot/test/plt/compare/%d.png"%batch, dpi=200)
    plt.close()


def plot_bivariate_gaussian3(mean, cov, ax, max_nstd=3, c='b'):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    for j in range(1, max_nstd+1):

        # Width and height are "full" widths, not radius
        width, height = 2 * j * np.sqrt(vals)
        ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, edgecolor=c, fill=False,
                          linewidth=1.0)

        ax.add_artist(ellip)
 
    return ellip


def Loss_Plot(train_batch_num, error_batch, loss_batch, file_name, x_axis_label,
              NLL_loss_batch=None, uncertainty_loss_batch=None):

    
    plt.subplot(2,1,1)
    plt.plot(train_batch_num, error_batch, 'b', linewidth=2.0, label="error")
    plt.ylabel("error")

    plt.subplot(2,1,2)
    plt.plot(train_batch_num, loss_batch, 'k', linewidth=2.0, label="loss")
    if NLL_loss_batch is not None:
        plt.plot(train_batch_num, NLL_loss_batch, 'g', linewidth=2.0, label="NLL_loss")
    if uncertainty_loss_batch is not None:
        plt.plot(train_batch_num, uncertainty_loss_batch, 'r', linewidth=2.0, label="uncertainty_loss")
    plt.xlabel(x_axis_label)
    plt.ylabel("loss")

    plt.savefig("Store_Results/plot/train/"+file_name+".png")
    plt.close()
 


def main():

    file_path_PCG = "Store_Results/plot/test/PCG/test_results.pkl"
    file_path_UAWPCG = "Store_Results/plot/test/UAWPCG/test_results.pkl"
    file_path = file_path_UAWPCG # file_path_PCG OR file_path_UAWPCG

   
    try:
        f = open(file_path, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path)

    results = pickle.load(f)

    print("====== The total number of data in the test set is: " + str(len(results)) + ' ========')

    for i  in [874]: # range(0, len(results), 10): # plotting the samples in the test set

        results_i = results[i]
        true_trajectories = results_i[0]
        pred_trajectories = results_i[1]
        PedsList_seq = results_i[2]
        lookup_seq = results_i[3]
        obs_length = results_i[5]
        dist_param_seq = results_i[6]
        true_trajectories_veh = results_i[7]
        VehsList_seq = results_i[8]
        lookup_seq_veh = results_i[9]
        grid_seq = results_i[10]
        grid_seq_veh = results_i[11]



        # covnert dist_param_seq to a torch tensor
        dist_param_seq = torch.from_numpy(dist_param_seq)
        
        plot_trajecotries(true_trajectories, pred_trajectories, obs_length,i,
                          dist_param_seq, PedsList_seq, lookup_seq, true_trajectories_veh,
                          VehsList_seq, lookup_seq_veh, grid_seq, grid_seq_veh)

if __name__ == '__main__':
    main()


