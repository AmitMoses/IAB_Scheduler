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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_split(dataset):
    """
    Split the dataset into train, validation and test.
    :param dataset: input dataset, which the iteration number is in the first dimension
    :return: train_dataset, valid_dataset, test_dataset
    """
    train_dataset = dataset[0:8000]
    valid_dataset = dataset[8000:9000]
    test_dataset = dataset[9000:10000]

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


def train(dataset_ue, dataset_iab, dataset_graph_iab, config, model):

    model.to(device)
    model.train(mode=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    train_loss, valid_loss, capacity_train_loss, capacity_valid_loss = [], [], [], []

    # === Split
    train_ue, valid_ue, _ = data_split(np.array(dataset_ue))
    train_iab, valid_iab, _ = data_split(np.array(dataset_iab))
    train_iab_graph, valid_iab_graph, _ = data_split(dataset_graph_iab)

    # Training process
    for epoch in range(config['epochs']):
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
        train_loader_iab_graph = DataLoader(train_iab_graph, batch_size=config['batch_size'])
        train_loader_ue_table = torch.utils.data.DataLoader(train_ue, batch_size=config['batch_size'])
        train_loader_iab_table = torch.utils.data.DataLoader(train_iab, batch_size=config['batch_size'])

        # === Iterate over all mini-batches
        # a = len(train_ue)
        # b = config['batch_size']
        for iab_data_graph, ue_data, iab_data in tqdm(zip(train_loader_iab_graph, train_loader_ue_table, train_loader_iab_table),
                                                      total=int((len(train_ue)/config['batch_size']))):
            # print(iab_data[0].x.shape)
            # print(ue_data[0].shape)

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

    # === Plots
    # Loss
    plt.figure()
    plt.title('LogLoss Curve \n minibatch_size = {} | learning_rate = {} | RegulationCost = {}'
              .format(config['batch_size'], config['learning_rate'], config['regulation_cost']))
    plt.semilogy(train_loss, label="Train")
    plt.semilogy(valid_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # Performance
    plt.figure()
    plt.title('LogLoss Curve \n minibatch_size = {} | learning_rate = {} | RegulationCost = {}'
              .format(config['batch_size'], config['learning_rate'], config['regulation_cost']))
    plt.semilogy(capacity_train_loss, label="Train")
    plt.semilogy(capacity_valid_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel('lossCapacity')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def main():
    main_path = '/common_space_docker/IAB_Scheduler'
    raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'
    path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'
    path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'

    UE_table_database, IAB_table_database, IAB_graph_database = \
        load_datasets(path_ue_table=path_UE,
                      path_iab_table=path_IAB,
                      raw_path_iab_graph=raw_paths_IAB_graph,
                      processed_path_iab_graph=processed_dir_IAB_graph)

    model_config = {
        'batch_size': 100,
        'epochs': 5,
        'learning_rate': 1e-5,
        'weight_decay': 0,
        'regulation_cost': 0
    }
    batch_v = [50, 100, 200]
    learn_v = [1e-3, 1e-4, 1e-5]
    for b in batch_v:
        for l in learn_v:
            print('minibatch_size = {} | learning_rate = {}'.format(b, l))
            model_config['batch_size'] = b
            model_config['learning_rate'] = l
            GCN_model = nnmod.ResourceAllocation_GNN()

            train(dataset_ue=UE_table_database,
                  dataset_iab=IAB_table_database,
                  dataset_graph_iab=IAB_graph_database,
                  config=model_config,
                  model=GCN_model)


if __name__ == '__main__':
    main()
