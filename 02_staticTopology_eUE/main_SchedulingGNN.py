__author__ = 'Amit'

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


    # === Split
    # print('before split')
    # print(len(dataset_graph_iab))
    # print(dataset_ue.shape)
    # print(dataset_iab.shape)

    train_ue, valid_ue, _ = datap.data_split(np.array(dataset_ue))
    train_iab, valid_iab, _ = datap.data_split(np.array(dataset_iab))
    train_iab_graph, valid_iab_graph, _ = datap.data_split(dataset_graph_iab)

    # Training process
    for epoch in range(config['epochs']):

        # if config['lr_change'] and (epoch > 60):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] / 100,
        #                                  weight_decay=config['weight_decay'])
        # elif config['lr_change'] and (epoch > 100):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] / 10,
        #                                  weight_decay=config['weight_decay'])

        # === Permutation
        p = np.random.permutation(len(train_iab_graph))
        train_ue = train_ue[p]
        train_iab = train_iab[p]
        train_iab_graph = [train_iab_graph[i] for i in p]

        p = np.random.permutation(len(valid_iab_graph))
        valid_ue = valid_ue[p]
        valid_iab = valid_iab[p]
        valid_iab_graph = [valid_iab_graph[i] for i in p]

        # === Batch division
        # print('before DataLoader')
        # print(len(train_iab_graph))
        # print(train_ue.shape)
        # print(train_iab.shape)

        train_loader_iab_graph = DataLoader(train_iab_graph, batch_size=config['batch_size'])
        train_loader_ue_table = torch.utils.data.DataLoader(train_ue, batch_size=config['batch_size'])
        train_loader_iab_table = torch.utils.data.DataLoader(train_iab, batch_size=config['batch_size'])


        # === Iterate over all mini-batches
        # a = len(train_ue)
        # b = config['batch_size']
        for iab_data_graph, ue_data, iab_data in tqdm(zip(train_loader_iab_graph, train_loader_ue_table, train_loader_iab_table),
                                                      total=int((len(train_ue)/config['batch_size']))):
            # print(iab_data[0].x.shape)

            # === Extract features for table datasets
            Train_UEbatch, Train_IABbatch, Train_UEidx = datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0,
                                                                         config['batch_size'])
            # === auxiliary label
            label_Train = datap.label_extractor(Train_UEbatch, Train_IABbatch)
            inputModel = torch.cat((Train_IABbatch, Train_UEbatch), dim=1)
            inputModel = inputModel.to(device)
            Train_UEidx = Train_UEidx.to(device)
            iab_data_graph = iab_data_graph.to(device)

            pred = model(inputModel, Train_UEidx, iab_data_graph)
            # === Compute the training loss and accuracy
            loss = datap.topology_cost(pred, label_Train, config['regulation_cost'])
            lossCapacity = datap.capacity_cost(pred, Train_UEbatch, Train_IABbatch)

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
        valid_loader_iab_graph = DataLoader(valid_iab_graph, batch_size=config['batch_size'])
        valid_loader_ue_table = torch.utils.data.DataLoader(valid_ue, batch_size=config['batch_size'])
        valid_loader_iab_table = torch.utils.data.DataLoader(valid_iab, batch_size=config['batch_size'])
        for iab_data_graph, ue_data, iab_data in zip(valid_loader_iab_graph, valid_loader_ue_table, valid_loader_iab_table):
            # === Extract features for table datasets
            Valid_UEbatch, Valid_IABbatch, Valid_UEidx = datap.get_batch(np.copy(ue_data), np.copy(iab_data), 0,
                                                                         config['batch_size'])
            # === auxiliary label
            label_Train = datap.label_extractor(Valid_UEbatch, Valid_IABbatch)
            inputModel = torch.cat((Valid_IABbatch, Valid_UEbatch), dim=1)
            inputModel = inputModel.to(device)
            Valid_UEidx = Valid_UEidx.to(device)
            iab_data_graph = iab_data_graph.to(device)

            pred_valid = model(inputModel, Valid_UEidx, iab_data_graph)
            # === Compute the training loss and accuracy
            validLoss = datap.topology_cost(pred_valid, label_Train, config['regulation_cost'])
            validLossCapacity = datap.capacity_cost(pred_valid, Valid_UEbatch, Valid_IABbatch)
            break

        print(
            "[Epoch]: %i, [Train Loss]: %.3E , [Train Capacity Loss]: %.6f Mbps | [Valid Loss]: %.3E , [Valid Capacity Loss]: %.6f Mbps"
            % (epoch + 1, loss.item(), lossCapacity, validLoss, validLossCapacity))

        train_loss.append(loss.item())
        capacity_train_loss.append(lossCapacity.detach().numpy())
        valid_loss.append(validLoss.detach().numpy())
        capacity_valid_loss.append(validLossCapacity.detach().numpy())

        # Save model
        # dir_path = '../common_space_docker/IAB_scheduler/saved_model/'
        dir_path = '../saved_model/'
        if config['if_save_model']:
            checkpoint_path = dir_path + str(config['save_model_path']) + '/epoch-{}.pt'
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
    plt.title('Performance Curve \n minibatch_size = {} | learning_rate = {} | weight_decay = {}'
              .format(config['batch_size'], config['learning_rate'], config['weight_decay']))
    plt.semilogy(capacity_train_loss, label="Train")
    plt.semilogy(capacity_valid_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel('lossCapacity')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def main():
    main_path = '/storage/amit/projects/IAB_Scheduler'
    # print('AAAAA')
    main_path = os.path.dirname(os.path.abspath(__file__))
    main_path = '../'
    # main_path = '/common_space_docker/IAB_Scheduler'

    raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'
    path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'

    UE_table_database, IAB_table_database, IAB_graph_database = \
        datap.load_datasets(path_ue_table=path_UE,
                            path_iab_table=path_IAB,
                            raw_path_iab_graph=raw_paths_IAB_graph,
                            processed_path_iab_graph=processed_dir_IAB_graph)

    print(UE_table_database)
    print(IAB_table_database)
    model_config = {
        'batch_size': 50,
        'epochs': 150,
        'learning_rate': 1e-3,
        'weight_decay': 0,
        'regulation_cost': 0,
        'lr_change': False,
        'if_save_model': True,
        'save_model_path': 'gnn_V3'
    }
    batch_v = [10]
    learn_v = [1e-4]
    wd_v = [1e-10]
    for b in batch_v:
        for l in learn_v:
            for w in wd_v:
                print('minibatch_size = {} | learning_rate = {} | weight_decay = {}'.format(b, l, w))
                model_config['batch_size'] = b
                model_config['learning_rate'] = l
                model_config['weight_decay'] = w
                GCN_model = nnmod.ResourceAllocation_GNN()

                train(dataset_ue=UE_table_database,
                      dataset_iab=IAB_table_database,
                      dataset_graph_iab=IAB_graph_database,
                      config=model_config,
                      model=GCN_model)


if __name__ == '__main__':
    main()

# minibatch_size = 10 | learning_rate = 0.0001
# [Epoch]: 96, [Train Loss]: 3.699E-07 , [Train Capacity Loss]: 1.543766 Mbps | [Valid Loss]: 6.354E-07 , [Valid Capacity Loss]: 2.373470 Mbps

# minibatch_size = 50 | learning_rate = 1e-3
# [Epoch]: 68, [Train Loss]: 1.112E-06 , [Train Capacity Loss]: 4.249455 Mbps | [Valid Loss]: 1.367E-06 , [Valid Capacity Loss]: 5.364619 Mbps

# minibatch_size = 10 | learning_rate = 0.0001 | weight_decay = 1e-10
# [Epoch]: 109, [Train Loss]: 1.897E-08 , [Train Capacity Loss]: 1.388600 Mbps | [Valid Loss]: 1.280E-07 , [Valid Capacity Loss]: 1.960467 Mbps

# gnn_V2
# minibatch_size = 10 | learning_rate = 0.0001 | weight_decay = 1e-10
# [Epoch]: 98, [Train Loss]: 3.408E-08 , [Train Capacity Loss]: 1.341962 Mbps | [Valid Loss]: 1.613E-07 , [Valid Capacity Loss]: 1.800668 Mbps
# [Epoch]: 110, [Train Loss]: 1.772E-07 , [Train Capacity Loss]: 1.050822 Mbps | [Valid Loss]: 1.022E-07 , [Valid Capacity Loss]: 1.334636 Mbps
# [Epoch]: 125, [Train Loss]: 3.939E-08 , [Train Capacity Loss]: 1.028648 Mbps | [Valid Loss]: 1.052E-06 , [Valid Capacity Loss]: 3.970101 Mbps


# gnn_V3
# [Epoch]: 82, [Train Loss]: 3.065E-07 , [Train Capacity Loss]: 1.622206 Mbps | [Valid Loss]: 3.079E-07 , [Valid Capacity Loss]: 2.641486 Mbps
# [Epoch]: 87, [Train Loss]: 2.871E-07 , [Train Capacity Loss]: 6.353583 Mbps | [Valid Loss]: 1.297E-08 , [Valid Capacity Loss]: 0.433673 Mbps
# [Epoch]: 91, [Train Loss]: 3.260E-07 , [Train Capacity Loss]: 2.212267 Mbps | [Valid Loss]: 5.957E-08 , [Valid Capacity Loss]: 0.680347 Mbps
# [Epoch]: 94, [Train Loss]: 9.189E-08 , [Train Capacity Loss]: 2.048471 Mbps | [Valid Loss]: 3.006E-07 , [Valid Capacity Loss]: 3.376941 Mbps
