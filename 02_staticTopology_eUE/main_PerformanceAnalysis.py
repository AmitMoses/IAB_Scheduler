__author__ = 'Amit'

import torch
import pandas as pd
import numpy as np
import NN_model as nnmod
import f_nnAnzlysis as nna
import f_SchedulingDataProcess as datap
import p_RadioParameters as rp
from torch_geometric.loader import DataLoader
import main_EDA as eda
import f_schedulers as scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Load Data
    # data_path = '../../database/ConstTopology/10000 samples'
    # data_path = '../database/DynamicTopology/10000 samples max 20/'
    # data_path = '../database/DynamicTopology/e6_m20_d2/'
    #
    # path_IAB = data_path + '/IAB_database.csv'
    # path_UE = data_path + '/UE_database.csv'
    #
    # IAB_database = pd.read_csv(path_IAB)
    # UE_database = pd.read_csv(path_UE)
    # test_UE = np.array(UE_database[9000:10000])
    # test_IAB = np.array(IAB_database[9000:10000])
    # print('one hop')
    # main_path = '../'
    # raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    # processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'
    # path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'
    # path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'

    print('multi-hop1')
    main_path = '../'
    raw_paths_IAB_graph = main_path + '/GraphDataset/data_v4/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data_v4/processed/'
    path_UE = main_path + '/database/DynamicTopology/data_v4/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/data_v4/IAB_database.csv'
    print(f'Total Bandwidth: {rp.Total_BW}')
    UE_table_database, IAB_table_database, IAB_graph_database = \
        datap.load_datasets(path_ue_table=path_UE,
                            path_iab_table=path_IAB,
                            raw_path_iab_graph=raw_paths_IAB_graph,
                            processed_path_iab_graph=processed_dir_IAB_graph)

    UE_table_rm_outlier, IAB_table_rm_outlier, IAB_graph_rm_outlier = \
        eda.remove_outlier_spectrum(UE_table_database, IAB_table_database, IAB_graph_database, isPlot=False)

    _, _, test_ue = datap.data_split(np.array(UE_table_rm_outlier), is_all=True)
    _, _, test_iab = datap.data_split(np.array(IAB_table_rm_outlier), is_all=True)
    _, _, test_iab_graph = datap.data_split(IAB_graph_rm_outlier, is_all=True)

    modelV0 = nna.load_model(nnmod.ResourceAllocation3DNN_v2(),
                          'DNN_V1', 150)

    # common data processing
    minibatch_size = test_ue.shape[0]
    # Compute the validation accuracy & loss ======================================================
    # === Batch division
    test_loader_iab_graph = DataLoader(test_iab_graph, batch_size=minibatch_size)
    test_loader_ue_table = torch.utils.data.DataLoader(test_ue, batch_size=minibatch_size)
    test_loader_iab_table = torch.utils.data.DataLoader(test_iab, batch_size=minibatch_size)
    for iab_data_graph, ue_data, iab_data in zip(test_loader_iab_graph, test_loader_ue_table, test_loader_iab_table):
        # === Extract features for table datasets
        Test_UEbatch, Test_IABbatch, Test_UEidx = \
            datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0, minibatch_size)
        # === auxiliary label
        label_Test = datap.label_extractor(Test_UEbatch, Test_IABbatch)
        inputModel = torch.cat((Test_IABbatch, Test_UEbatch), dim=1)
        inputModel = inputModel.to(device)
        Test_UEidx = Test_UEidx.to(device)
        iab_data_graph = iab_data_graph.to(device)

        # # model prediction
        # test_pred = modelV0(inputModel, Test_UEidx, iab_data_graph)
        # test_pred = modelV0(inputModel, Test_UEidx)
        # test_pred = nna.simple_resource_allocation(test_ue, test_iab, iab_div=0.5)
        # test_pred = scheduler.equal_resource(test_ue, test_iab)
        # test_pred = scheduler.out_of_band_like(test_ue, test_iab)
        # test_pred = scheduler.fair_access_n_backhaul(test_ue, test_iab, iab_div=0.5)
        test_pred = scheduler.optimal(test_ue, test_iab)
    # ==========================================================================


        # label extractor
        UE_pred = test_pred[:, :, :40]  # removes IABs
        IAB_pred = test_pred[:, :, 40:42]  # removes UEs
        UE_efficiency, UE_capacity = datap.input_extract_for_cost(Test_UEbatch)
        IAB_efficiency, IAB_capacity = datap.input_extract_for_cost(Test_IABbatch)
        IAB_capacity[:, -1, :] = 0
        efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
        capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)

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

        break


if __name__ == '__main__':
    main()
