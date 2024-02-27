
import argparse
import pickle
from utils.helper import *

def main():
    
    parser = argparse.ArgumentParser() 

    # method selection
    parser.add_argument('--method', type=int, default=5,
                            help='Method you want to display its test result  \
                            (4 = PCG, 5 = UAW-PCG ') 
    # Minimum acceptalbe distance between two pedestrians
    parser.add_argument('--D_min', type=int, default=0.7, 
                        help='Minimum distance for which the TTC is calculated')
    # Minimum acceptalbe distance between a pedstrian and a vehicle
    parser.add_argument('--D_min_veh', type=int, default=1.0, 
                        help='Minimum distance for which the TTC is calculated')
    args = parser.parse_args()


    file_path_PCG = "Store_Results/plot/test/PCG/test_results.pkl"
    file_path_UAWPCG = "Store_Results/plot/test/UAWPCG/test_results.pkl"

    if args.method == 4:
        file_path = file_path_PCG
        print("====== PCG results ======")
    elif args.method == 5:
        file_path = file_path_UAWPCG
        print("====== UAW-PCG results ======")
    else:
        raise ValueError("Invalid method number")
    
    try:
        f = open(file_path, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path)

    results = pickle.load(f)

    ave_error = []
    final_error = []
    MHD_error = []
    speed_error = []
    heading_error = []
    num_coll_homo = []
    all_num_cases_homo = []
    num_coll_hetero = []
    all_num_cases_hetero = []
    NLL_list = []
    ESV_sigma1 = []
    ESV_sigma2 = []
    ESV_sigma3 = []
    data_point_num = 0

    for i in range(len(results)): # each i is one sample or batch (since batch_size is 1 during test)

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

        pred_trajectories = torch.from_numpy(pred_trajectories)
        true_trajectories = torch.from_numpy(true_trajectories)
        true_trajectories_veh = torch.from_numpy(true_trajectories_veh)
        dist_param_seq = torch.from_numpy(dist_param_seq)


        
        PedsList_obs = sum(PedsList_seq[:obs_length], [])
        error_batch = get_mean_error(pred_trajectories[obs_length:,:,:2], true_trajectories[obs_length:,:,:2],
                                      PedsList_seq[obs_length:], PedsList_obs, False, lookup_seq)
        final_error_batch = get_final_error(pred_trajectories[obs_length:,:,:2], true_trajectories[obs_length:,:,:2],
                                             PedsList_seq[obs_length:],PedsList_obs, lookup_seq)
        MHD_error_batch = get_hausdorff_distance(pred_trajectories[obs_length:,:,:2], true_trajectories[obs_length:,:,:2],
                                                  PedsList_seq[obs_length:], PedsList_obs, False, lookup_seq)
        speed_error_batch, heading_error_batch = get_velocity_errors(pred_trajectories[obs_length:,:,2:4],
                                                                      true_trajectories[obs_length:,:,2:4], 
                                                                      PedsList_seq[obs_length:], PedsList_obs, 
                                                                      False, lookup_seq)
        num_coll_homo_batch, all_num_cases_homo_batch = get_num_collisions_homo(pred_trajectories[obs_length:,:,:2],
                                                                                 PedsList_seq[obs_length:], PedsList_obs, 
                                                                                 False, lookup_seq, args.D_min)
        num_coll_hetero_batch, all_num_cases_hetero_batch = get_num_collisions_hetero(pred_trajectories[obs_length:,:,:2],
                                                                                       PedsList_seq[obs_length:], PedsList_obs,
                                                                                        False, lookup_seq,
                                                                                        true_trajectories_veh[obs_length:,:,:2],
                                                                                        VehsList_seq[obs_length:], lookup_seq_veh,
                                                                                        args.D_min_veh)

        NLL_list_batch, _ = get_mean_NLL(dist_param_seq[obs_length:,:,:],
                                            true_trajectories[obs_length:,:,:2],
                                            PedsList_seq[obs_length:], 
                                            PedsList_obs, False, lookup_seq)
                
        ESV_sigma1_batch, ESV_sigma2_batch, ESV_sigma3_batch, counter = Delta_Empirical_Sigma_Value(
                                                                                            dist_param_seq[obs_length:,:,:],
                                                                                            true_trajectories[obs_length:,:,:2],
                                                                                            PedsList_seq[obs_length:], PedsList_obs,
                                                                                            False, lookup_seq)

        ave_error.append(error_batch)
        final_error.append(final_error_batch)
        MHD_error.append(MHD_error_batch)
        speed_error.append(speed_error_batch)
        heading_error.append(heading_error_batch)
       
        num_coll_homo.append(num_coll_homo_batch)
        all_num_cases_homo.append(all_num_cases_homo_batch)
        num_coll_hetero.append(num_coll_hetero_batch)
        all_num_cases_hetero.append(all_num_cases_hetero_batch)

        NLL_list.extend(NLL_list_batch)
        ESV_sigma1.append(ESV_sigma1_batch)
        ESV_sigma2.append(ESV_sigma2_batch)
        ESV_sigma3.append(ESV_sigma3_batch)
        data_point_num += counter

    ADE = sum(ave_error).item() / len(results)
    FDE = sum(final_error).item() / len(results)
    MHD = sum(MHD_error).item() / len(results)
    SE = sum(speed_error).item() / len(results)
    HE = sum(heading_error).item() /len(results)
    Collision = (sum(num_coll_homo)+sum(num_coll_hetero))/(sum(all_num_cases_homo)+sum(all_num_cases_hetero)) * 100
    Collision_pedped = sum(num_coll_homo)/sum(all_num_cases_homo) * 100
    Collision_pedveh = sum(num_coll_hetero)/sum(all_num_cases_hetero) * 100
    
    NLL_mean, NLL_std = calculate_mean_and_std(NLL_list)
    sigma1 = (sum(ESV_sigma1) / data_point_num)-0.39
    sigma2 = (sum(ESV_sigma2) / data_point_num)-0.86
    sigma3 = (sum(ESV_sigma3) / data_point_num)-0.99


    print('Average displacement error (ADE): ', round(ADE,4)) 
    print('Final displacement error (FDE): ', round(FDE, 4))
    print('Hausdorff distance error (MHD): ', round(MHD, 4))
    # print('Speed error (SE): ', round(SE, 4))
    # print('Average heading error (HE): ', round(HE, 2))
    # print('Overll percentage of collision:', round(Collision, 4))
    # print('Percentage of collision between pedestrians: ', round(Collision_pedped, 4))
    # print('Percentage of collision between pedestrians and vehicles:', round(Collision_pedveh, 4))
    print('NLL: ', round(NLL_mean,4), '±', round(NLL_std,4))
    print('ESV_sigma1: ', round(sigma1,4))
    print('ESV_sigma2: ', round(sigma2,4))
    print('ESV_sigma3: ', round(sigma3,4))


if __name__ == '__main__':
    main()


