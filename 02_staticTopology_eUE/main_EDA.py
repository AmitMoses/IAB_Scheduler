__author__ = 'Amit'

import numpy as np
import f_SchedulingDataProcess as datap
import torch
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object


def hist_plot(data, title, x_label, y_label):
    length = data.shape[0]
    labels = list(range(1, length + 1))
    x = np.arange(len(labels))  # the label locations
    fig = plt.figure(figsize=(25, 7))
    ax = fig.add_subplot(111)
    ax.bar(x, data, 1, label=y_label)
    # plot details
    ax.set_xlim(0, length + 1)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    # ticks
    labels_t = list(range(0, length, 50))
    x_t = np.array(labels_t)
    ax.set_xticks(x_t)
    ax.set_xticklabels(labels_t)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)
    plt.show()


def get_cap_eff_data(UE_table_database, IAB_table_database):
    Test_UEbatch, Test_IABbatch, Test_UEidx = \
        datap.get_batch(np.copy(UE_table_database), np.copy(IAB_table_database), 0, len(UE_table_database))

    UE_efficiency, UE_capacity = datap.input_extract_for_cost(Test_UEbatch)
    IAB_efficiency, IAB_capacity = datap.input_extract_for_cost(Test_IABbatch)
    IAB_capacity[:, -1, :] = 0
    efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
    capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)

    return capacity, efficiency


def remove_outlier_spectrum(UE_table_database, IAB_table_database, IAB_graph_database, isPlot = True):
    capacity, efficiency = get_cap_eff_data(UE_table_database, IAB_table_database)

    bw_link = capacity / efficiency
    bw_iteration = torch.sum(bw_link, dim=(1, 2))
    spectrum_outliers = 520
    outlier_idx = torch.where(bw_iteration > spectrum_outliers)[0]
    # print(outlier_idx)
    # print(len(outlier_idx))

    UE_data_rm = UE_table_database.drop(index=outlier_idx)
    IAB_data_rm = IAB_table_database.drop(index=outlier_idx)
    # IAB_graph_rm = IAB_graph_database.pop(outlier_idx[0])
    IAB_graph_rm = delete_multiple_element(IAB_graph_database, outlier_idx)
    # print(len(IAB_graph_rm))
    # print(UE_data_rm)

    if isPlot:
        # outlier box plot
        fig1, ax1 = plt.subplots()
        ax1.set_title('Requested BW')
        ax1.boxplot(bw_iteration, vert=False)
        ax1.minorticks_on()
        plt.grid()
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()

    return UE_data_rm, IAB_data_rm, IAB_graph_rm


def main():
    main_path = '../'
    # raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    # processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'
    # path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'
    # path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'

    raw_paths_IAB_graph = main_path + '/GraphDataset/data_v4/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data_v4/processed/'
    path_UE = main_path + '/database/DynamicTopology/data_v4/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/data_v4/IAB_database.csv'

    UE_table_database, IAB_table_database, IAB_graph_database = \
        datap.load_datasets(path_ue_table=path_UE,
                            path_iab_table=path_IAB,
                            raw_path_iab_graph=raw_paths_IAB_graph,
                            processed_path_iab_graph=processed_dir_IAB_graph,
                            is_generate=False)

    print(UE_table_database)
    print(UE_table_database.shape)
    print(IAB_table_database)
    print(IAB_table_database.shape)

    for i in range(0, 5):
        data = IAB_graph_database[i]
        g = torch_geometric.utils.to_networkx(data)
        plt.figure()
        nx.draw(g)
        plt.show()

    capacity, efficiency = get_cap_eff_data(UE_table_database, IAB_table_database)
    tot_capacity = torch.sum(capacity, dim=(1, 2))
    mean_efficiency = torch.mean(efficiency, dim=(1, 2))
    print(f'Average Capacity: {torch.mean(tot_capacity)}')
    print(f'Average Efficiency: {torch.mean(mean_efficiency)}')

    # UE_data_rm, IAB_data_rm, IAB_graph_rm = \
    #     remove_outlier_spectrum(UE_table_database, IAB_table_database,IAB_graph_database, isPlot=True)
    # print(UE_data_rm)
    # print(IAB_data_rm)



    # # Split dataset
    # train_ue, valid_ue, test_ue = datap.data_split(np.array(UE_data_rm), is_all=True)
    # tarin_iab, valid_iab, test_iab = datap.data_split(np.array(IAB_data_rm), is_all=True)
    # train_iab_graph, test_iab_graph, test_iab_graph = datap.data_split(IAB_graph_rm, is_all=True)




if __name__ == '__main__':
    main()