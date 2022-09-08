__author__ = 'Amit'

import numpy as np
import torch
import p_RadioParameters as rp
import pandas as pd
import sys
sys.path.insert(1, '../GraphDataset/')
import DataGenerator as data_gen
from sklearn.model_selection import train_test_split
import random


def input_extract_for_cost(model_input):
    """
    This function get the model input (including batchs inputs) and return the
    the efficiency & capacity of every link in the shape of the model output:
    [batch_size, IAB_num, data_len*] = [batch_size, 10, data_len*]

    Input:    model_input - tensor that contaion features on the UE/IAB, there
                            are 4 features (Downlink Mbps demand, Downlink CQI,
                            Uplink Mbps demand, Uplink CQI)
                            in shape [batch_size, features * UE_num or** features * IAB]
                            Note: the input is the output of 'get_batch' function.
    Output:   capacity_batch_mat - Data on the links capacity demand in tensor
                                   shape [batch_size, 10, 2 or*** 20]
              efficiency_batch_mat - Data on the links efficiency in tensor shape
                                     [batch_size, 10, 2 or*** 20]

    *   - depend on the input database (UE\IAB), the concatenate over the two output
          will be in the exact output shape [batch_size, 10, 22]
    **  - depend on the input database (UE\IAB), 400 for UE and 40 for IAB
    *** - depend on the input database (UE\IAB), for UE: UE_per_IAB * UL&DL_links
                                                          = 10 * 2 = 20
                                                 for IAB: IAB_perIAB * UL&DL_links
                                                          = 1 * 2 = 20
    """
    # Parameters:
    batch_size = model_input.shape[0]
    data_len = int(
        model_input.shape[1] / (rp.IAB_num * 3))  # 2 because just half the data is neccesery (efficiency or CQI)
    # initialize
    efficiency_batch_mat = torch.zeros(batch_size, rp.IAB_num, data_len)
    capacity_batch_mat = torch.zeros(batch_size, rp.IAB_num, data_len)

    # Extract capacity & efficiency data from every input in the batch
    for i in range(0, batch_size):
        # input shape
        one_batch_input = model_input[i]
        one_batch_input = one_batch_input.reshape(-1, rp.feature_num)
        # capacity extract from input
        capacity = torch.cat((one_batch_input[:, 0:1], one_batch_input[:, 3:4]), dim=1)
        capacity = torch.reshape(capacity, (rp.IAB_num, -1))

        # efficiency extract from input
        efficiency = torch.cat((one_batch_input[:, 1:2], one_batch_input[:, 4:5]), dim=1)
        efficiency = torch.reshape(efficiency, (rp.IAB_num, -1))
        # efficiency calculate from CQI & CQI2efficiency dict
        # efficiency = torch.tensor([[rp.CQI2efficiency[int(allocation)] for allocation in IAB_out] for IAB_out in CQI])
        # efficiency = CQI

        # # efficiency extract from input
        # efficiency = torch.cat((one_batch_input[:, 1:2], one_batch_input[:, 3:4]), dim=1)
        # efficiency = torch.reshape(efficiency, (10, -1))

        # save resultes
        capacity_batch_mat[i, :, :] = capacity
        efficiency_batch_mat[i, :, :] = efficiency
    return efficiency_batch_mat, capacity_batch_mat


def label_extractor(Data_UEbatch, Data_IABbatch):
    """
    NEED TO CHANGE LABEL FOR IAB-DONOR AKA IAB10 BACKHAUL TO ZERO BW:
    currently getting the label from UL & DL of the users (can be changing by
    changing the database)
    """
    # IAB_num = 10
    UE_efficiency, UE_capacity = input_extract_for_cost(Data_UEbatch)
    IAB_efficiency, IAB_capacity = input_extract_for_cost(Data_IABbatch)
    IAB_capacity[:, -1, :] = 0
    # print(IAB_capacity)
    efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
    capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)
    # efficiency_index = torch.where(efficiency == 0)  # find index where efficiency == 0
    # efficiency[efficiency_index] = rp.eps  # replies efficiency == 0 in eps. Prevents division by zero
    label = (capacity / efficiency)
    label = (label / rp.Total_BW) * 1e6  # [MHZ]
    return label


def usher(UEs_data, sample_number):
    """
    This function was made in order to split the allocation problem into a sub-problem that each IAB will have to be
    solved separately (based on the same NN model).
    :param UEs_data:        UE database as numpy
    :param sample_number:   The sample number in the database. each index point to specific data sample
    :return:
    """
    # number_of_UEs = 100
    info_cells_per_UE = 5
    in_vec = UEs_data[sample_number]
    in_mat = in_vec.reshape(
        (rp.UE_num, info_cells_per_UE))  # reshape to matrix, for output: each row represents different UE
    in_iab = in_mat[:, 0]  # get IAB info only
    sorted_args_by_iab = np.argsort(in_iab)  # get indexes of how it should be sorted

    sorted_mat = np.zeros_like(in_mat)

    for idx, arg in enumerate(sorted_args_by_iab):
        sorted_mat[idx] = in_mat[arg]
    sorted_vec = sorted_mat.reshape((1, -1))  # this reshape is unnecessary
    return sorted_args_by_iab, sorted_vec, sorted_mat


def cqi2eff_in_matrix_col(mat, col):
    eff = torch.tensor([[rp.CQI2efficiency[int(allocation)] for allocation in mat[:, col]]])
    efficiency_index = torch.where(eff == 0)  # find index where efficiency == 0
    eff[efficiency_index] = rp.eps  # replies efficiency == 0 in eps. Prevents division by zero
    bw = mat[:, int(col-1)] / eff
    mat[:, col] = eff
    # mat[:, int(col-1)] = bw
    return mat


def bw_extract_from_mat(mat, col):
    eff = torch.tensor([[rp.CQI2efficiency[int(allocation)] for allocation in mat[:, col]]])
    efficiency_index = torch.where(eff == 0)  # find index where efficiency == 0
    eff[efficiency_index] = rp.eps  # replies efficiency == 0 in eps. Prevents division by zero
    bw = mat[:, int(col-1)] / eff
    return bw


def add_label_feature(mat, cqi_col_1, cqi_col_2):
    bw_1 = bw_extract_from_mat(mat, cqi_col_1)
    bw_2 = bw_extract_from_mat(mat, cqi_col_2)
    mat = np.array(mat)
    insert_idx_1 = 3
    insert_ixx_2 = 5 + 1
    mat = np.insert(mat, insert_idx_1, bw_1, axis=1)
    mat = np.append(mat, bw_2.T, axis=1)
    mat = torch.tensor(mat)
    return mat


def get_batch(UE_db, IAB_db, r_min, r_max, is_noise=False):
    """
    take a desirable number of samples from the two databases and rearrange them (with the help from the usher function)
     to match the input of the NN model.
    :param is_noise:
    :param UE_db:
    :param IAB_db:
    :param r_min:
    :param r_max:
    :return:
    """
    # # Database permutation:
    # p = np.random.permutation(len(UE_db))
    # UE_db = UE_db[p]
    # IAB_db = IAB_db[p]

    # Parameters:
    # UE_num = 100
    # IAB_num = 10
    # maxUEperBS = 11
    # feature_num = 4

    # initialize:
    UE_batch_ = torch.zeros((r_max - r_min, rp.IAB_num * rp.maxUEperBS * rp.feature_num))
    IAB_batch_ = torch.zeros((r_max - r_min, rp.IAB_num * rp.feature_num))
    ueNumInIAB_ = torch.zeros((r_max - r_min, rp.IAB_num))
    IABNumReq_ = torch.zeros((r_max - r_min, rp.IAB_num))
    # create batch:
    # print(UE_db)
    # print(UE_db.shape)
    for index, i in enumerate(range(r_min, r_max)):
        _, _, UE_mat_ = usher(UE_db, i)
        if is_noise:
            UE_mat_ = add_cqi_noise(UE_mat_)
            UE_mat_ = add_rate_noise(UE_mat_)
        # Extended UE input to match maximum of 'maxUEperBS' UE per IAB
        UE_mat_extend_ = np.zeros((rp.IAB_num * rp.maxUEperBS, 5))
        line_index = 0
        ue_counter = 0
        for i_ue, _ in enumerate(UE_mat_extend_):
            # Check if the loop go to the next IAB, in this case start counting UE from 0 (new IAB)
            if 101 + (i_ue // rp.maxUEperBS) > UE_mat_extend_[i_ue-1, 0]:
                ue_counter = 0
            UE_mat_extend_[i_ue, 0] = 101 + (i_ue // rp.maxUEperBS)  # set IAB number in UE_mat_extend
            # If there is data in the UE of the corresponding IAB (in the line) - copy to UE_mat_extend
            if UE_mat_[line_index, 0] == UE_mat_extend_[i_ue, 0]:
                UE_mat_extend_[i_ue, 1:] = UE_mat_[line_index, 1:]
                line_index = line_index + 1
                ue_counter = ue_counter + 1
                ueNumInIAB_[index, int((i_ue // rp.maxUEperBS))] = ue_counter
                if line_index == rp.UE_num:
                    break

        # return UE_mat_extend
        # eff = torch.tensor([[rp.CQI2efficiency[int(allocation)] for allocation in UE_mat_extend[:, 2]]])
        # UE_mat_extend[:, 2] = eff
        mat_temp = add_label_feature(np.copy(UE_mat_extend_), 2, 4)
        mat_temp = cqi2eff_in_matrix_col(mat_temp, 2)
        mat_temp = cqi2eff_in_matrix_col(mat_temp, 5)

        # UE_mat_extend_ = cqi2eff_in_matrix_col(UE_mat_extend_, 2)
        # UE_mat_extend_ = cqi2eff_in_matrix_col(UE_mat_extend_, 4)
        # return UE_mat_extend

        UE_mat_input = mat_temp[:, 1:].reshape(-1)
        UE_batch_[index, :] = UE_mat_input

        IAB_mat_ = IAB_db[i]
        IAB_mat_ = IAB_mat_.reshape((-1, 5))
        if is_noise:
            IAB_mat_ = add_cqi_noise(IAB_mat_)
            IAB_mat_ = add_rate_noise(IAB_mat_)
        IAB_mat_temp = add_label_feature(np.copy(IAB_mat_), 2, 4)
        IAB_mat_temp = cqi2eff_in_matrix_col(IAB_mat_temp, 2)
        IAB_mat_temp = cqi2eff_in_matrix_col(IAB_mat_temp, 5)
        IAB_mat_input_ = IAB_mat_temp[:, 1:].reshape(-1)
        IAB_batch_[index, :] = IAB_mat_input_

    # Generate UE index Vector
    UeIndexVec_ = torch.zeros((r_max - r_min, rp.IAB_num, rp.maxUEperBS))
    # UeIndexVec_re_ = torch.zeros((r_max - r_min, rp.IAB_num, rp.maxUEperBS*2))
    for lineIdx, line in enumerate(ueNumInIAB_):
        for numIdx, num in enumerate(line):
            UeIndexVec_[lineIdx, numIdx, 0:int(num)] = 1

    UeIndexVec_re_ = torch.repeat_interleave(UeIndexVec_, 2, dim=2)
    # print(UeIndexVec[0, 0, :])
    # print(UeIndexVec_re[0, 0, :])
    return UE_batch_, IAB_batch_, UeIndexVec_re_


def add_cqi_noise(table):
    table_noise = np.copy(table)
    noise_bin = np.random.binomial(n=4, p=0.5, size=len(table_noise[:, 2]))
    noise = noise_bin - np.round(np.mean(noise_bin))
    table_noise[:, 2] = np.clip(table_noise[:, 2] + noise, 1, 15)
    noise_bin = np.random.binomial(n=4, p=0.5, size=len(table_noise[:, 2]))
    noise = noise_bin - np.round(np.mean(noise_bin))
    table_noise[:, 4] = np.clip(table_noise[:, 2] + noise, 1, 15)
    return table_noise

def add_rate_noise(table):
    table_noise = np.copy(table)
    table_noise[:, 1] = np.max(table_noise[:, 1] + np.random.normal(0, 5, len(table_noise[:, 1])), 0)
    table_noise[:, 3] = np.max(table_noise[:, 3] + np.random.normal(0, 5, len(table_noise[:, 3])), 0)
    return table_noise


def topology_cost(output, label, Regulation=0):
    """
    input size [batch_size,10,22] for both inputs
    """
    cost = (output - label)
    index = torch.where(cost > 0)
    cost[index] = Regulation * cost[index]
    cost = torch.abs(cost)
    cost = torch.sum(cost, dim=(1, 2))
    cost = torch.mean(cost)
    return cost


def capacity_cost(output, Data_UEbatch, Data_IABbatch):
    UE_efficiency, UE_capacity = input_extract_for_cost(Data_UEbatch)
    IAB_efficiency, IAB_capacity = input_extract_for_cost(Data_IABbatch)
    efficiency = torch.cat((UE_efficiency, IAB_efficiency), dim=2)
    capacity = torch.cat((UE_capacity, IAB_capacity), dim=2)
    CapacityCost = capacity - output * rp.Total_BW * efficiency / 1e6
    index = torch.where(CapacityCost < 0)
    CapacityCost[index] = 0
    CapacityCost = torch.sum(CapacityCost, dim=(1, 2))
    CapacityCost = torch.mean(CapacityCost)

    BW_cost = capacity/efficiency - output * rp.Total_BW / 1e6
    BW_cost[index] = 0
    BW_cost = torch.sum(BW_cost, dim=(1, 2))
    BW_cost = torch.mean(BW_cost)

    return CapacityCost, BW_cost


def data_split(dataset, is_all=True, type='', print_info=False):
    """
    Split the dataset into train, validation and test.
    :param type: data type
    :param is_all: choose if take all data or part of
    :param dataset: input dataset, which the iteration number is in the first dimension
    :return: train_dataset, valid_dataset, test_dataset
    """
    if not is_all:
        dataset = dataset[0:1000]

    train_dataset, val_test = train_test_split(dataset, test_size=0.2, shuffle=False)
    valid_dataset, test_dataset = train_test_split(val_test, test_size=0.5, shuffle=False)

    if print_info:
        print(f'Dataset division - {type}:')
        print(f'Train length:       {len(train_dataset)}')
        print(f'Validation length:  {len(valid_dataset)}')
        print(f'Test length:        {len(test_dataset)}')

    return train_dataset, valid_dataset, test_dataset


def load_datasets(path_ue_table, path_iab_table, raw_path_iab_graph, processed_path_iab_graph, is_generate=False):
    """
    :param is_generate: load for process the graph dataset
    :param path_ue_table: directory path for UE table dataset
    :param path_iab_table: directory path for IAB table dataset
    :param raw_path_iab_graph: directory path for UE PyTorch Geometric (graph) dataset
    :param processed_path_iab_graph: directory path for IAB PyTorch Geometric (graph) dataset
    :return: load the datasets
    """
    UE_table_database = pd.read_csv(path_ue_table)
    IAB_table_database = pd.read_csv(path_iab_table)

    if is_generate:
        IAB_graph_database = data_gen.process(raw_path_iab_graph, processed_path_iab_graph)
    else:
        IAB_graph_database = data_gen.load(processed_path_iab_graph)

    return UE_table_database, IAB_table_database, IAB_graph_database