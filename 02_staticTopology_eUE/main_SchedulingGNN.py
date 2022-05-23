__author__ = 'Amit'

from typing import List

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(dataset_ue, dataset_iab, dataset_graph_iab, config, model):

    # print('train function input')
    # print(len(dataset_graph_iab))
    # print(dataset_ue.shape)
    # print(dataset_iab.shape)

    model.to(device)
    model.train(mode=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    train_loss, valid_loss, capacity_train_loss, capacity_valid_loss = [], [], [], []
    bw_train_loss, bw_valid_loss = [], []

    # # === Split
    # print('before split')
    # print(len(dataset_graph_iab))
    # print(dataset_ue.shape)
    # print(dataset_iab.shape)

    train_ue, valid_ue, _ = datap.data_split(np.array(dataset_ue), is_all=True, type='UE')
    train_iab, valid_iab, _ = datap.data_split(np.array(dataset_iab), is_all=True, type='IAB')
    train_iab_graph, valid_iab_graph, _ = datap.data_split(dataset_graph_iab, is_all=True, type='IAB-graph')

    # Training process
    for epoch in range(config['epochs']):

        if config['lr_change'] and (epoch > 100):
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] / 100,
                                         weight_decay=config['weight_decay'])
        elif config['lr_change'] and (epoch > 50):
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] / 10,
                                         weight_decay=config['weight_decay'])



        # === Permutation
        p = np.random.permutation(len(train_iab_graph))
        train_ue = train_ue[p]
        train_iab = train_iab[p]
        train_iab_graph = [train_iab_graph[i] for i in p]

        p = np.random.permutation(len(valid_iab_graph))
        valid_ue = valid_ue[p]
        valid_iab = valid_iab[p]
        valid_iab_graph = [valid_iab_graph[i] for i in p]

        # # === Batch division
        # print('before DataLoader')
        # print(len(train_iab_graph))
        # print(train_ue.shape)
        # print(train_iab.shape)
        train_loader_iab_graph = DataLoader(train_iab_graph, batch_size=config['batch_size'], drop_last=True)
        train_loader_ue_table = torch.utils.data.DataLoader(train_ue, batch_size=config['batch_size'], drop_last=True)
        train_loader_iab_table = torch.utils.data.DataLoader(train_iab, batch_size=config['batch_size'], drop_last=True)


        # === Iterate over all mini-batches
        for iab_data_graph, ue_data, iab_data in tqdm(zip(train_loader_iab_graph, train_loader_ue_table, train_loader_iab_table),
                                                      total=int((len(train_ue)/config['batch_size']))):

            # === Extract features for table datasets
            Train_UEbatch, Train_IABbatch, Train_UEidx = datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0,
                                                                         config['batch_size'])
            # === auxiliary label
            label_Train = datap.label_extractor(Train_UEbatch, Train_IABbatch)
            inputModel = torch.cat((Train_IABbatch, Train_UEbatch), dim=1)
            inputModel = inputModel.to(device)
            Train_UEidx = Train_UEidx.to(device)
            iab_data_graph = iab_data_graph.to(device)

            # pred = model(inputModel, Train_UEidx, iab_data_graph)
            pred = model(inputModel, Train_UEidx)

            # === Compute the training loss and accuracy
            loss = datap.topology_cost(pred, label_Train, config['regulation_cost'])
            lossCapacity, lossBW = datap.capacity_cost(pred, Train_UEbatch, Train_IABbatch)

            # === zero the gradients before running
            # the backward pass.
            optimizer.zero_grad()

            # === Backward pass to compute the gradient
            # of loss w.r.t our learnable params.
            loss.backward()

            # === Update params
            optimizer.step()

        # Compute the validation accuracy & loss
        # === Batch division
        valid_loader_iab_graph = DataLoader(valid_iab_graph, batch_size=len(valid_ue))
        valid_loader_ue_table = torch.utils.data.DataLoader(valid_ue, batch_size=len(valid_ue))
        valid_loader_iab_table = torch.utils.data.DataLoader(valid_iab, batch_size=len(valid_ue))
        for iab_data_graph, ue_data, iab_data in zip(valid_loader_iab_graph, valid_loader_ue_table, valid_loader_iab_table):
            # === Extract features for table datasets
            Valid_UEbatch, Valid_IABbatch, Valid_UEidx = datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0,
                                                                         len(valid_ue))
            # === auxiliary label
            label_Train = datap.label_extractor(Valid_UEbatch, Valid_IABbatch)
            inputModel = torch.cat((Valid_IABbatch, Valid_UEbatch), dim=1)
            inputModel = inputModel.to(device)
            Valid_UEidx = Valid_UEidx.to(device)
            iab_data_graph = iab_data_graph.to(device)
            # pred_valid = model(inputModel, Valid_UEidx, iab_data_graph)
            pred_valid = model(inputModel, Valid_UEidx)
            # === Compute the training loss and accuracy
            validLoss = datap.topology_cost(pred_valid, label_Train, config['regulation_cost'])
            validLossCapacity, validLossBW = datap.capacity_cost(pred_valid, Valid_UEbatch, Valid_IABbatch)
            break

        # Save model
        # dir_path = '../common_space_docker/IAB_scheduler/saved_model/'
        dir_path = '../saved_model/'
        if config['if_save_model']:
            checkpoint_path = dir_path + str(config['save_model_dir']) + '/epoch-{}.pt'
            torch.save(model.state_dict(), checkpoint_path.format(epoch + 1))


        # print(
        #     "[Epoch]: %i, [Train Loss]: %.3E , [Train Capacity Loss]: %.6f Mbps , [Train BW loss]: %.6f MHz | "
        #     "[Valid Loss]: %.3E , [Valid Capacity Loss]: %.6f Mbps, [Valid BW loss]: %.6f MHz"
        #     % (epoch + 1, loss.item(), lossCapacity, lossBW, validLoss, validLossCapacity, validLossBW))

        print(
            "[Epoch]: %i, [Train Loss]: %.3E , [Train Capacity Loss]: %.6f Mbps | "
            "[Valid Loss]: %.3E , [Valid Capacity Loss]: %.6f Mbps"
            % (epoch + 1, loss.item(), lossCapacity, validLoss, validLossCapacity))

        train_loss.append(loss.item())
        capacity_train_loss.append(lossCapacity.detach().numpy())
        valid_loss.append(validLoss.detach().numpy())
        capacity_valid_loss.append(validLossCapacity.detach().numpy())
        bw_train_loss.append(lossBW.detach().numpy())
        bw_valid_loss.append(validLossBW.detach().numpy())

        # Save model
        # dir_path = '../common_space_docker/IAB_scheduler/saved_model/'
        dir_path = '../saved_model/'
        if config['if_save_model']:
            checkpoint_path = dir_path + str(config['save_model_dir']) + '/epoch-{}.pt'
            # checkpoint_path = '/saved_models/' + str(directory) + '/epoch-{}.pt'
            torch.save(model.state_dict(), checkpoint_path.format(epoch + 1))

    # === Plots
    # Loss
    # plt.figure()
    # plt.title('LogLoss Curve \n minibatch_size = {} | learning_rate = {} | RegulationCost = {}'
    #           .format(config['batch_size'], config['learning_rate'], config['regulation_cost']))
    # plt.semilogy(train_loss, label="Train")
    # plt.semilogy(valid_loss, label="Validation")
    # plt.xlabel("Epoch")
    # plt.ylabel('Loss')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()

    # Performance
    plt.figure()
    plt.title('Performance Curve \n minibatch_size = {} | learning_rate = {} | weight_decay = {} \n R-cost = {}, l_chance = {}'
              .format(config['batch_size'], config['learning_rate'], config['weight_decay'], config['regulation_cost'], config['lr_change']))
    plt.semilogy(capacity_train_loss, label="Train")
    plt.semilogy(capacity_valid_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel('lossCapacity')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # plt.figure()
    # plt.title(
    #     'Performance Curve \n minibatch_size = {} | learning_rate = {} | weight_decay = {} \n R-cost = {}, l_chance = {}'
    #     .format(config['batch_size'], config['learning_rate'], config['weight_decay'], config['regulation_cost'],
    #             config['lr_change']))
    # plt.semilogy(bw_train_loss, label="Train")
    # plt.semilogy(bw_valid_loss, label="Validation")
    # plt.xlabel("Epoch")
    # plt.ylabel('Loss Bandwidth')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()


def main():

    main_path = '../'


    # raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    # processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'
    # path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'
    # path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'

    # raw_paths_IAB_graph = main_path + '/GraphDataset/data_v2/raw/'
    # processed_dir_IAB_graph = main_path + '/GraphDataset/data_v2/processed/'
    # path_UE = main_path + '/database/DynamicTopology/data_v3/UE_database.csv'
    # path_IAB = main_path + '/database/DynamicTopology/data_v3/IAB_database.csv'

    raw_paths_IAB_graph = main_path + '/GraphDataset/data_v4/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data_v4/processed/'
    path_UE = main_path + '/database/DynamicTopology/data_v4/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/data_v4/IAB_database.csv'

    UE_table_database, IAB_table_database, IAB_graph_database = \
        datap.load_datasets(path_ue_table=path_UE,
                            path_iab_table=path_IAB,
                            raw_path_iab_graph=raw_paths_IAB_graph,
                            processed_path_iab_graph=processed_dir_IAB_graph)
    print(IAB_table_database)
    print('k11')
    UE_table_rm_outlier, IAB_table_rm_outlier, IAB_graph_rm_outlier = \
        eda.remove_outlier_spectrum(UE_table_database, IAB_table_database, IAB_graph_database, isPlot=False)
    # UE_table_rm_outlier, IAB_table_rm_outlier, IAB_graph_rm_outlier = UE_table_database,IAB_table_database,IAB_graph_database

    model_config = {
        'batch_size': 50,
        'epochs': 150,
        'learning_rate': 1e-3,
        'weight_decay': 0,
        'regulation_cost': 1e-3,
        'lr_change': True,
        'if_save_model': False,
        'save_model_dir': 'DNN_V1'
    }

    lr_change_v = [True]
    learn_v = [1e-3, 1e-4]
    wd_v = [1e-6, 0]
    regulation_cost_v = [1e-1, 1e-3]
    batch_v = [50, 10, 5]

    for l_c in lr_change_v:
        for l in learn_v:
            for w in wd_v:
                for rc in regulation_cost_v:
                    for b in batch_v:
                        # print('minibatch_size = {} | learning_rate = {} | weight_decay = {} | regulation_cost = {}'
                        #       .format(b, l, w, i
                        #       rc))
                        model_config['batch_size'] = b
                        model_config['learning_rate'] = l
                        model_config['weight_decay'] = w
                        model_config['regulation_cost'] = rc
                        model_config['lr_change'] = l_c
                        NNmodel = nnmod.ResourceAllocation3DNN_v2()
                        # NNmodel = nnmod.ResourceAllocation_GCNConv()
                        print(NNmodel)
                        print(model_config)
                        train(dataset_ue=UE_table_rm_outlier,
                              dataset_iab=IAB_table_rm_outlier,
                              dataset_graph_iab=IAB_graph_rm_outlier,
                              config=model_config,
                              model=NNmodel)


if __name__ == '__main__':
    main()


# gnn_V1
# minibatch_size = 10 | learning_rate = 0.0001 | weight_decay = 1e-10

# ResourceAllocationDynamicGelu_V1
# DNN_V1
# [Epoch]: 128, [Train Loss]: 6.924E-04 , [Train Capacity Loss]: 0.424863 Mbps | [Valid Loss]: 1.750E-03 , [Valid Capacity Loss]: 1.874989 Mbps
# Average Difference 9.348701 [Mbps] Average Unfulfilled Links: 7.837949

# DNN_V2
# ResourceAllocationDynamicGelu2
#     lr_change_v = [True]
#     learn_v = [1e-4] 50 -> 150
#     wd_v = [0]
#     regulation_cost_v = [1e-3]
#     batch_v = [5]
# [Epoch]: 74, [Train Loss]: 3.717E-04 , [Train Capacity Loss]: 2.995756 Mbps , [Train BW loss]: 0.966975 MHz | [Valid Loss]: 3.835E-04 , [Valid Capacity Loss]: 0.573994 Mbps, [Valid BW loss]: 0.210195 MHz
# [Epoch]: 88, [Train Loss]: 4.428E-04 , [Train Capacity Loss]: 0.000000 Mbps , [Train BW loss]: 0.000000 MHz | [Valid Loss]: 9.705E-04 , [Valid Capacity Loss]: 2.101021 Mbps, [Valid BW loss]: 0.337864 MHz
# [Epoch]: 113, [Train Loss]: 4.107E-04 , [Train Capacity Loss]: 0.363695 Mbps , [Train BW loss]: 0.109471 MHz | [Valid Loss]: 4.222E-04 , [Valid Capacity Loss]: 0.000000 Mbps, [Valid BW loss]: 0.000000 MHz
# [Epoch]: 124, [Train Loss]: 4.348E-04 , [Train Capacity Loss]: 0.000000 Mbps , [Train BW loss]: 0.000000 MHz | [Valid Loss]: 3.978E-04 , [Valid Capacity Loss]: 0.306384 Mbps, [Valid BW loss]: 0.083432 MHz
# [Epoch]: 149, [Train Loss]: 3.590E-04 , [Train Capacity Loss]: 0.134178 Mbps , [Train BW loss]: 0.040387 MHz | [Valid Loss]: 2.470E-03 , [Valid Capacity Loss]: 9.158083 Mbps, [Valid BW loss]: 1.762604 MHz
# Average Difference 5.8439975 [Mbps]
# Average Unfulfilled Links: 1.894359

# DNN_V3
#     learn_v = [1e-4] 50 -> 100
#     wd_v = [0]
#     regulation_cost_v = [1e-3]
#     batch_v = [5]
# ResourceAllocation3DNN
# [Epoch]: 94, [Train Loss]: 8.429E-04 , [Train Capacity Loss]: 0.807634 Mbps | [Valid Loss]: 2.728E-03 , [Valid Capacity Loss]: 9.231317 Mbps
# [Epoch]: 101, [Train Loss]: 4.177E-04 , [Train Capacity Loss]: 0.000000 Mbps | [Valid Loss]: 2.746E-03 , [Valid Capacity Loss]: 9.300915 Mbps
# [Epoch]: 138, [Train Loss]: 4.379E-04 , [Train Capacity Loss]: 0.049632 Mbps | [Valid Loss]: 2.883E-03 , [Valid Capacity Loss]: 9.962917 Mbps
# [Epoch]: 150, [Train Loss]: 3.676E-04 , [Train Capacity Loss]: 2.433138 Mbps | [Valid Loss]: 2.746E-03 , [Valid Capacity Loss]: 9.581235 Mbps


# ResourceAllocation3DNN_v2
# DNN_V1
# [Epoch]: 92, [Train Loss]: 3.707E-04 , [Train Capacity Loss]: 0.049657 Mbps | [Valid Loss]: 3.038E-03 , [Valid Capacity Loss]: 9.454494 Mbps
# [Epoch]: 142, [Train Loss]: 4.185E-04 , [Train Capacity Loss]: 0.434011 Mbps | [Valid Loss]: 3.131E-03 , [Valid Capacity Loss]: 9.364378 Mbps
# [Epoch]: 150, [Train Loss]: 4.536E-04 , [Train Capacity Loss]: 0.600925 Mbps | [Valid Loss]: 3.214E-03 , [Valid Capacity Loss]: 9.712662 Mbps