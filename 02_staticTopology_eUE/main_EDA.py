__author__ = 'Amit'

import numpy as np
import f_SchedulingDataProcess as datap
import torch
import matplotlib.pyplot as plt


def main():
    main_path = '../'
    # raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    # processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'
    # path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'
    # path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'

    raw_paths_IAB_graph = main_path + '/GraphDataset/data_v2/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data_v2/processed/'
    path_UE = main_path + '/database/DynamicTopology/data_v2/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/data_v2/IAB_database.csv'

    UE_table_database, IAB_table_database, IAB_graph_database = \
        datap.load_datasets(path_ue_table=path_UE,
                            path_iab_table=path_IAB,
                            raw_path_iab_graph=raw_paths_IAB_graph,
                            processed_path_iab_graph=processed_dir_IAB_graph)

    train_ue, valid_ue, test_ue = datap.data_split(np.array(UE_table_database), is_all=True)
    tarin_iab, valid_iab, test_iab = datap.data_split(np.array(IAB_table_database), is_all=True)
    train_iab_graph, test_iab_graph, test_iab_graph = datap.data_split(IAB_graph_database, is_all=True)

    Test_UEbatch, Test_IABbatch, Test_UEidx = \
        datap.get_batch(np.copy(test_ue), np.copy(test_iab), 0, len(test_ue))


    UE_efficiency, UE_capacity = datap.input_extract_for_cost(Test_UEbatch)
    IAB_efficiency, IAB_capacity = datap.input_extract_for_cost(Test_IABbatch)
    IAB_capacity[:, -1, :] = 0
    efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
    capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)

    tot_capacity = torch.sum(capacity, dim=(1, 2))
    mean_efficiency = torch.mean(efficiency, dim=(1, 2))
    cap_size = tot_capacity.shape[0]
    eff_size = mean_efficiency.shape[0]


    # Capacity plot
    labels = list(range(1, cap_size+1))
    x = np.arange(len(labels))  # the label locations
    fig = plt.figure(figsize=(25, 7))
    ax = fig.add_subplot(111)
    ax.bar(x, tot_capacity, 1, label='Capacity')
    # plot details
    ax.set_xlim(0, cap_size+1)
    ax.set_ylabel('Capacity [Mbps]')
    ax.set_xlabel('Sample')
    ax.set_title('Total Requested Capacity')
    # ticks
    labels_t = list(range(0, cap_size, 50))
    x_t = np.array(labels_t)
    ax.set_xticks(x_t)
    ax.set_xticklabels(labels_t)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)
    plt.show()

    # effeciency plot
    labels = list(range(1, eff_size+1))
    x = np.arange(len(labels))  # the label locations
    fig = plt.figure(figsize=(25, 7))
    ax = fig.add_subplot(111)
    ax.bar(x, mean_efficiency, 1, label='efficiency')
    # plot details
    ax.set_xlim(0, eff_size+1)
    ax.set_ylabel('Efficiency [Mbps/sec/Hz]')
    ax.set_xlabel('Sample')
    ax.set_title('Efficiency Mean over all links')
    # ticks
    labels_t = list(range(0, eff_size, 50))
    x_t = np.array(labels_t)
    ax.set_xticks(x_t)
    ax.set_xticklabels(labels_t)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)
    plt.show()

    print(f'Average Capacity: {torch.mean(tot_capacity)}')
    print(f'Average Efficiency: {torch.mean(mean_efficiency)}')

if __name__ == '__main__':
    main()