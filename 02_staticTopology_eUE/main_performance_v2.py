import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import NN_model as nnmod
import f_SchedulingDataProcess as datap
import p_RadioParameters as rp
from tqdm import tqdm
import sys
sys.path.insert(1, '../GraphDataset/')
import DataGenerator as data_gen
from torch_geometric.loader import DataLoader
import os
import main_EDA as eda
import f_schedulers as scheduler
import f_nnAnzlysis as nna


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    main_path = '../'
    raw_paths_IAB_graph = main_path + '/GraphDataset/data_v4/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data_v4/processed/'
    path_UE = main_path + '/database/DynamicTopology/data_v4/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/data_v4/IAB_database.csv'

    UE_table_database, IAB_table_database, IAB_graph_database = \
        datap.load_datasets(path_ue_table=path_UE,
                            path_iab_table=path_IAB,
                            raw_path_iab_graph=raw_paths_IAB_graph,
                            processed_path_iab_graph=processed_dir_IAB_graph)

    UE_table_rm_outlier, IAB_table_rm_outlier, IAB_graph_rm_outlier = \
        eda.remove_outlier_spectrum(UE_table_database, IAB_table_database, IAB_graph_database, isPlot=False)

    modelV0 = scheduler.load_model(nnmod.ResourceAllocation3DNN_v4(),
                                   'DNN_noise_V2', 17)
    # modelV0 = scheduler.load_model(nnmod.ResourceAllocation3DNN_v2(),
    #                                'DNN_V1', 100)

    train_ue, valid_ue, test_ue = datap.data_split(np.array(UE_table_rm_outlier), is_all=True, type='UE')
    train_iab, valid_iab, test_iab = datap.data_split(np.array(IAB_table_rm_outlier), is_all=True, type='IAB')
    train_iab_graph, valid_iab_graph, test_iab_graph = datap.data_split(IAB_graph_rm_outlier,
                                                                        is_all=True, type='IAB-graph')

    minibatch_size = test_ue.shape[0]
    test_loader_iab_graph = DataLoader(test_iab_graph, batch_size=minibatch_size, drop_last=True)
    test_loader_ue_table = torch.utils.data.DataLoader(test_ue, batch_size=minibatch_size, drop_last=True)
    test_loader_iab_table = torch.utils.data.DataLoader(test_iab, batch_size=minibatch_size, drop_last=True)

    for iab_data_graph, ue_data, iab_data in tqdm(
            zip(test_loader_iab_graph, test_loader_ue_table, test_loader_iab_table),
            total=int((len(test_ue) / minibatch_size))):
        # === Extract features for table datasets
        Test_UEbatch, Test_IABbatch, Test_UEidx = datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0,
                                                                     minibatch_size)
        Test_UEbatch_noise, Test_IABbatch_noise, _ = datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0,
                                                                       minibatch_size, is_noise=False)
        # === auxiliary label
        label_Train = datap.label_extractor(Test_UEbatch, Test_IABbatch)
        # inputModel = torch.cat((Train_IABbatch, Train_UEbatch), dim=1)
        inputModel = torch.cat((Test_IABbatch_noise, Test_UEbatch_noise), dim=1)
        inputModel = inputModel.to(device)
        Test_UEidx = Test_UEidx.to(device)
        iab_data_graph = iab_data_graph.to(device)

        # pred = model(inputModel, Train_UEidx, iab_data_graph)
        # pred = modelV0(inputModel, Test_UEidx)

        # === Compute the training loss and accuracy
        # lossCapacity, lossBW = datap.capacity_cost(pred, Test_UEbatch, Test_IABbatch)
        # print(lossCapacity)

        # schedulers = [scheduler.fair_access_n_backhaul]
        # schedulers_list = ['FnA']
        schedulers = [scheduler.equal_resource, scheduler.split_spectrum, scheduler.split_spectrum_backhaul_aware,
                      scheduler.fair_access_n_backhaul, modelV0, scheduler.optimal]
        schedulers_list = ['ERA', 'SS', 'SS-BA', 'FAnB', 'DNN', 'Centralize']

        Average_Allocation_Ability = []
        Average_Difference, Average_Difference_ue, Average_Difference_iab = [], [], []
        Average_Unfulfilled_UE_Links, Average_Unfulfilled_IAB_Links = [], []
        Bandwidth_Usage = []
        Network_efficiency = []

        Average_Allocation_Ability_v = []
        Average_Difference_v, Average_Difference_ue_v, Average_Difference_iab_v = [], [], []
        Average_Unfulfilled_UE_Links_v, Average_Unfulfilled_IAB_Links_v = [], []
        Bandwidth_Usage_v = []
        Network_efficiency_v = []

        for method in schedulers:
            if method == modelV0:
                # method.eval()
                test_pred = modelV0(inputModel, Test_UEidx)
            # elif method == modelV1:
            #     test_pred = modelV1(inputModel, Test_UEidx, iab_data_graph)
            else:
                test_pred = method(Test_UEbatch_noise, Test_IABbatch_noise, minibatch_size)

            UE_pred = test_pred[:, :, :40]  # removes IABs
            IAB_pred = test_pred[:, :, 40:42]  # removes UEs
            UE_efficiency, UE_capacity = datap.input_extract_for_cost(Test_UEbatch)
            IAB_efficiency, IAB_capacity = datap.input_extract_for_cost(Test_IABbatch)
            IAB_capacity[:, -1, :] = 0
            efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
            capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)

            # BW usage
            BW_usage = nna.bandwidth_usage(capacity, efficiency, test_pred)

            # System efficiency
            sys_eff = nna.network_efficiency(capacity, efficiency, test_pred)

            cap, capCost = nna.cap_req_vs_alloc_bars(capacity, efficiency, test_pred, is_plot=False)
            print(np.mean(capCost))

            # ue only
            cap_ue, capCost_ue = nna.cap_req_vs_alloc_bars(UE_capacity, UE_efficiency, UE_pred, is_plot=False)
            # iab only
            cap_iab, capCost_iab = nna.cap_req_vs_alloc_bars(IAB_capacity, IAB_efficiency, IAB_pred, is_plot=False)

            # Bars plot: Average unfulfilled links
            # Average unfulfilled links - UE
            ue_unfil_links = nna.unfulfilled_links_bars(UE_capacity, UE_efficiency, UE_pred, is_plot=False)
            # Average unfulfilled links - IAB
            iab_unfil_links = nna.unfulfilled_links_bars(IAB_capacity, IAB_efficiency, IAB_pred, is_plot=False,
                                                         is_access=False)

            # Scores
            print(method)
            print('Average Allocation Ability:      ', nna.allocation_ability(cap, capCost), '%')
            print('Average Difference:              ', np.mean(capCost), '[Mbps]')
            print('Average Difference - UE:         ', np.mean(capCost_ue), '[Mbps]')
            print('Average Difference - IAB:        ', np.mean(capCost_iab), '[Mbps]')
            print('Average Unfulfilled UE Links:    ', np.mean(ue_unfil_links))
            print('Average Unfulfilled IAB Links:   ', np.mean(iab_unfil_links))
            print('Bandwidth Usage:                 ', float(torch.mean(BW_usage)), '[MHz]')
            print('Network efficiency:              ', float(torch.mean(sys_eff)), '[Mpbs/Hz]')
            print('--------------------------------------------------')

            Average_Allocation_Ability.append(nna.allocation_ability(cap, capCost))
            Average_Difference.append(np.mean(capCost))
            Average_Difference_ue.append(np.mean(capCost_ue))
            Average_Difference_iab.append(np.mean(capCost_iab))
            Average_Unfulfilled_UE_Links.append(np.mean(ue_unfil_links))
            Average_Unfulfilled_IAB_Links.append(np.mean(iab_unfil_links))
            Bandwidth_Usage.append(float(torch.mean(BW_usage)))
            Network_efficiency.append(float(torch.mean(sys_eff)))

            Average_Difference_v.append(capCost)
            Average_Difference_ue_v.append(capCost_ue)
            Average_Difference_iab_v.append(capCost_iab)
            Average_Unfulfilled_UE_Links_v.append(ue_unfil_links)
            Average_Unfulfilled_IAB_Links_v.append(iab_unfil_links)
            Bandwidth_Usage_v.append(BW_usage.detach().numpy())
            Network_efficiency_v.append(sys_eff.detach().numpy())


    # Plots

    # Average_Allocation_Ability
    nna.draw_bar_plot(x=schedulers_list,
                      y=Average_Allocation_Ability,
                      title='Average Allocation Ability',
                      x_label='[%]',
                      y_label='Method',
                      is_save=True)

    # Average Difference
    nna.seaborn_box_plot(x=schedulers_list,
                         y=Average_Difference_v,
                         title='Average Performance',
                         x_label='[Mbps]',
                         y_label='Method',
                         is_save=True)

    # # Average Difference UE
    # nna.draw_bar_plot(x=schedulers_list,
    #                   y=Average_Difference_ue,
    #                   title='Average Performance - Access',
    #                   x_label='[Mbps]',
    #                   y_label='Method',
    #                   is_save=True)
    #
    # # Average Difference IAB
    # nna.draw_bar_plot(x=schedulers_list,
    #                   y=Average_Difference_iab,
    #                   title='Average Performance - Backhaul',
    #                   x_label='[Mbps]',
    #                   y_label='Method',
    #                   is_save=True)
    #
    # # Average Unfulfilled UE Links
    # nna.draw_bar_plot(x=schedulers_list,
    #                  y=Average_Unfulfilled_UE_Links,
    #                  title='Average Unfulfilled UE Links',
    #                  x_label='[#]',
    #                  y_label='Method',
    #                  is_save=True)
    #
    # # Average Unfulfilled IAB Links
    # nna.draw_bar_plot(x=schedulers_list,
    #                  y=Average_Unfulfilled_IAB_Links,
    #                  title='Average Unfulfilled IAB Links',
    #                  x_label='[#]',
    #                  y_label='Method',
    #                   is_save=True)
    #
    # # Average Loss per link - Access
    # ue_loos_per_link = nna.capacity_loss_per_link(Average_Unfulfilled_UE_Links, Average_Difference_ue)
    # nna.draw_bar_plot(x=schedulers_list,
    #                   y=ue_loos_per_link,
    #                   title='Average Loss per link - Access',
    #                   x_label='[Mbps]',
    #                   y_label='Method',
    #                   is_save=True)
    #
    # # Average Loss per link - Backhaul
    # iab_loos_per_link = nna.capacity_loss_per_link(Average_Unfulfilled_IAB_Links, Average_Difference_iab)
    # nna.draw_bar_plot(x=schedulers_list,
    #                   y=iab_loos_per_link,
    #                   title='Average Loss per link - Backhaul',
    #                   x_label='[Mbps]',
    #                   y_label='Method',
    #                   is_save=True)
    #
    # # Bandwidth Usage
    # nna.seaborn_box_plot(x=schedulers_list,
    #                   y=Bandwidth_Usage_v,
    #                   title='Bandwidth Usage',
    #                   x_label='[MHz]',
    #                   y_label='Method',
    #                   is_save=True)


if __name__ == '__main__':
    main()