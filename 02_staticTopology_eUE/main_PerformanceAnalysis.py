__author__ = 'Amit'

import torch
import pandas as pd
import numpy as np
import NN_model as nnmod
import f_nnAnzlysis as nna
import f_SchedulingDataProcess as datap
import p_RadioParameters as rp
import main_SchedulingProject_2 as cost

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Load Data
    # data_path = '../../database/ConstTopology/10000 samples'
    # data_path = '../database/DynamicTopology/10000 samples max 20/'
    data_path = '../database/DynamicTopology/e6_m20_d2/'

    path_IAB = data_path + '/IAB_database.csv'
    path_UE = data_path + '/UE_database.csv'

    IAB_database = pd.read_csv(path_IAB)
    UE_database = pd.read_csv(path_UE)
    test_UE = np.array(UE_database[9000:10000])
    test_IAB = np.array(IAB_database[9000:10000])

    modelV0 = nna.load_model(nnmod.ResourceAllocationDynamicGelu(),
                          'S02_model_V3', 45)

    # common data processing
    minibatch_size = test_UE.shape[0]
    test_UEbatch, test_IABbatch, test_UEIdx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    test_UEbatch_n, test_IABbatch_n, test_UEIdx_n = datap.get_batch_new(np.copy(test_UE), np.copy(test_IAB), 0, minibatch_size)
    input_val = torch.cat((test_IABbatch_n, test_UEbatch_n), dim=1)
    input_val = input_val.to(device)
    test_UEIdx = test_UEIdx.to(device)

    # model prediction
    test_pred = modelV0(input_val, test_UEIdx)
    # test_pred = nna.simple_resource_allocation(test_UE, test_IAB, iab_div=0.5)

    # label extractor
    UE_pred = test_pred[:, :, :40]  # removes IABs
    IAB_pred = test_pred[:, :, 40:42]  # removes UEs
    UE_efficiency, UE_capacity = datap.input_extract_for_cost(test_UEbatch)
    IAB_efficiency, IAB_capacity = datap.input_extract_for_cost(test_IABbatch)
    IAB_capacity[:, -1, :] = 0
    efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
    capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)

    # # Boxplot: Resource allocation per IAB
    # # data processing
    # req_vec = IAB_capacity.detach().numpy()
    # out_IAB = IAB_pred * IAB_efficiency * rp.Total_BW / 1e6
    # rel_vec = out_IAB.detach().numpy()
    # # plot
    # nna.allocat_per_IAB_boxplot(req_vec, rel_vec)

    # Bars plot: Capacity (UL+DL): Requested vs Allocation
    # plot
    cap, capCost = nna.cap_req_vs_alloc_bars(capacity, efficiency, test_pred)

    # Bars plot: Average unfulfilled links
    # Average unfulfilled links
    unfil_links = nna.unfulfilled_links_bars(UE_capacity, UE_efficiency, UE_pred)

    # Scores
    print('Average Allocation Ability', nna.allocation_ability(cap, capCost), '%')
    print('Average Difference', np.mean(capCost), '[Mbps]')
    print('Average Unfulfilled Links:', np.mean(unfil_links))



if __name__ == '__main__':
    main()
