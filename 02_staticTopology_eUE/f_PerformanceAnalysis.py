__author__ = 'Amit'

import pandas as pd
import numpy as np
import torch
import NN_model as nnmod
import f_nnAnzlysis as perfa
import f_SchedulingDataProcess as datap
import p_RadioParameters as rp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    data_path = '../../database/DynamicTopology/10000 samples max 20'

    path_IAB = data_path + '/IAB_database.csv'
    path_UE = data_path + '/UE_database.csv'

    IAB_database = pd.read_csv(path_IAB)
    UE_database = pd.read_csv(path_UE)

    test_UE = np.array(UE_database[9000:10000])
    test_IAB = np.array(IAB_database[9000:10000])

    modelV0 = perfa.load_model(nnmod.resourceAllocation_Dynamic(), 'S02_model_V2', 109)

    # common data processing
    minibatch_size = test_UE.shape[0]
    test_UEbatch, test_IABbatch, Test_UEidx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    input_val = torch.cat((test_IABbatch, test_UEbatch), dim=1)
    input_val = input_val.to(device)
    Test_UEidx = Test_UEidx.to(device)
    test_pred = modelV0(input_val, Test_UEidx)
    UE_pred = test_pred[:, :, :rp.maxUEperBS*2]  # removes IABs
    IAB_pred = test_pred[:, :, (rp.maxUEperBS*2):((rp.maxUEperBS+rp.backhaul_num)*2)]  # removes UEs
    UE_efficiency, UE_capacity = datap.input_extract_for_cost(test_UEbatch)
    IAB_efficiency, IAB_capacity = datap.input_extract_for_cost(test_IABbatch)
    IAB_capacity[:, -1, :] = 0
    efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
    capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)

    # Simple allocation
    test_simple_all = perfa.simple_resource_allocation(test_UE, test_IAB, 0.5)
    UE_simple_all = test_simple_all[:, :, :rp.maxUEperBS * 2]  # removes IABs
    # # Boxplot: Resource allocation per IAB
    # # data processing
    # req_vec = IAB_capacity.detach().numpy()
    # out_IAB = IAB_pred * IAB_efficiency * rp.Total_BW / 1e6
    # rel_vec = out_IAB.detach().numpy()
    # # plot model
    # perfa.allocat_per_iab_boxplot(req_vec, rel_vec)

    # Bars plot: Capacity (UL+DL): Requested vs Allocation
    capa, CapacityCost = perfa.cap_req_vs_alloc_bars(capacity, efficiency, test_pred)    # prediction
    capa_simple_all, CapacityCost_simple_all = perfa.cap_req_vs_alloc_bars(capacity, efficiency, test_simple_all)    # simple allocation

    # # Bars plot: Average unfulfilled links
    unfil_links = perfa.unfulfilled_links_bars(UE_capacity, UE_efficiency, UE_pred)
    unfil_links_simple_all = perfa.unfulfilled_links_bars(UE_capacity, UE_efficiency, UE_simple_all)


    # # Scores
    print('Model prediction scores:')
    print(' Average Allocation Ability: ', perfa.allocation_ability(capa, CapacityCost), '%')
    print(' Average Difference:         ', np.mean(capa - CapacityCost), '[Mbps]')
    print(' Average Unfulfilled Links:  ', np.mean(unfil_links), '[# of UE]')

    print('\nSimple allocation scores:')
    print(' Average Allocation Ability: ', perfa.allocation_ability(capa_simple_all, CapacityCost_simple_all), '%')
    print(' Average Difference:         ', np.mean(capa_simple_all - CapacityCost_simple_all), '[Mbps]')
    print(' Average Unfulfilled Links:  ', np.mean(unfil_links_simple_all), '[# of UE]')


if __name__ == '__main__':
    main()
