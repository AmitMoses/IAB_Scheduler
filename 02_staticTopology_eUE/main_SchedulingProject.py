__author__ = 'Amit'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import NN_model as nnmod
import f_SchedulingDataProcess as datap
import p_RadioParameters as rp
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(TrainData_IAB, TrainData_UE, ValidData_IAB, ValidData_UE, model, minibatch_size, epoch, learning_rate,
                wd, save_model=False, directory='2FCN', lr_change=False, RegulationCost=0):
    model.to(device)
    model.train(mode=True)
    # model.cpu()

    # we use the optim package to apply
    # SGD for our parameter updates
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # built-in L2
    # Adam for our parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)  # built-in L2

    train_loss, valid_loss, capacity_train_loss, capacity_valid_loss = [], [], [], []
    epochs = []

    # Training
    for t in range(epoch):
        # Divide data into mini batches
        if lr_change and (t > 120):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 100, weight_decay=wd)
        elif lr_change and (t > 60):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 10, weight_decay=wd)

        p = np.random.permutation(len(TrainData_UE))
        TrainData_UE = TrainData_UE[p]
        TrainData_IAB = TrainData_IAB[p]
        p = np.random.permutation(len(ValidData_UE))
        ValidData_UE = ValidData_UE[p]
        ValidData_IAB = ValidData_IAB[p]

        for i in tqdm(range(0, TrainData_UE.shape[0] - minibatch_size, minibatch_size)):
            # Feed forward to get the logits
            Train_UEbatch, Train_IABbatch, Train_UEidx = datap.get_batch(TrainData_UE, TrainData_IAB, i,
                                                                         i + minibatch_size)
            Train_UEbatch_n, Train_IABbatch_n, Train_UEidx_n = datap.get_batch_new(np.copy(TrainData_UE), np.copy(TrainData_IAB), i,
                                                                                   i + minibatch_size)

            label_Train = datap.label_extractor(Train_UEbatch, Train_IABbatch)
            inputModel = torch.cat((Train_IABbatch_n, Train_UEbatch_n), dim=1)

            # temp = label_Train_norm.view(minibatch_size, -1)
            # inputModel = temp

            inputModel = inputModel.to(device)
            Train_UEidx = Train_UEidx.to(device)
            # inputModel = inputModel.cpu()

            pred = model(inputModel, Train_UEidx)
            # Compute the training loss and accuracy
            loss = datap.topology_cost(pred, label_Train, RegulationCost)
            lossCapacity = datap.capacity_cost(pred, Train_UEbatch, Train_IABbatch)

            # zero the gradients before running
            # the backward pass.
            optimizer.zero_grad()

            # Backward pass to compute the gradient
            # of loss w.r.t our learnable params.
            loss.backward()

            # Update params
            optimizer.step()

        # Compute the validation accuracy & loss
        Valid_UEbatch, Valid_IABbatch, Valid_UEidx = datap.get_batch(ValidData_UE, ValidData_IAB, 0, minibatch_size)
        Valid_UEbatch_n, Valid_IABbatch_n, Valid_UEidx_n = datap.get_batch_new(np.copy(ValidData_UE), np.copy(ValidData_IAB), 0, minibatch_size)
        label_Valid = datap.label_extractor(Valid_UEbatch, Valid_IABbatch)
        input_val = torch.cat((Valid_IABbatch_n, Valid_UEbatch_n), dim=1)

        # temp_val = label_Valid_norm.view(minibatch_size, -1)
        # input_val = temp_val

        input_val = input_val.to(device)
        Valid_UEidx = Valid_UEidx.to(device)
        # input_val = input_val.cpu()
        valid_pred = model(input_val, Valid_UEidx)
        validLoss = datap.topology_cost(valid_pred, label_Valid, RegulationCost)
        validLossCapacity = datap.capacity_cost(valid_pred, Valid_UEbatch, Valid_IABbatch)

        # Save model
        # dir_path = '../common_space_docker/IAB_scheduler/saved_model/'
        dir_path = '../saved_model/'
        if save_model:
            checkpoint_path = dir_path + str(directory) + '/epoch-{}.pt'
            # checkpoint_path = '/saved_models/' + str(directory) + '/epoch-{}.pt'
            torch.save(model.state_dict(), checkpoint_path.format(t + 1))

        print(
            "[Epoch]: %i, [Train Loss]: %.3E , [Train Capacity Loss]: %.6f Mbps | [Valid Loss] %.3E , [Valid Capacity "
            "Loss]: %.6f Mbps "
            % (t + 1, loss.item(), lossCapacity, validLoss, validLossCapacity))
        # display.clear_output(wait=True)

        # Save error on each epoch
        epochs.append(t)
        train_loss.append(loss.item())
        valid_loss.append(validLoss.detach().numpy())
        capacity_train_loss.append(lossCapacity.detach().numpy())
        capacity_valid_loss.append(validLossCapacity.detach().numpy())

    # plotting

    # Loss
    plt.figure()
    plt.title('LogLoss Curve \n minibatch_size = {} | learning_rate = {} | RegulationCost = {}'
              .format(minibatch_size, learning_rate, RegulationCost))
    plt.semilogy(epochs, train_loss, label="Train")
    plt.semilogy(epochs, valid_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()

    if save_model:
        checkpoint_path = dir_path + str(directory) + '/Loss_Plot.jpg'
        plt.savefig(checkpoint_path)
    plt.show()

    # Capacity Loss
    plt.figure()
    plt.title('Capacity LogLoss Curve \n minibatch_size = {} | learning_rate = {} | RegulationCost = {}'
              .format(minibatch_size, learning_rate, RegulationCost))
    plt.semilogy(epochs, capacity_train_loss, label="Train")
    plt.semilogy(epochs, capacity_valid_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.grid()

    if save_model:
        checkpoint_path = dir_path + str(directory) + '/Capacity_Loss_Plot.jpg'
        plt.savefig(checkpoint_path)
    plt.show()


def main():
    # Load Data
    # data_path = '../../database/ConstTopology/10000 samples'
    # data_path = '../../database/DynamicTopology/1000 samples'
    # data_path = '../database/DynamicTopology/10000 samples max 20'
    data_path = '../database/DynamicTopology/e6_m20_d3'
    path_IAB = data_path + '/IAB_database.csv'
    path_UE = data_path + '/UE_database.csv'

    IAB_database = pd.read_csv(path_IAB)
    UE_database = pd.read_csv(path_UE)


    train_UE = np.array(UE_database[0:800])
    train_IAB = np.array(IAB_database[0:800])
    valid_UE = np.array(UE_database[800:900])
    valid_IAB = np.array(IAB_database[800:900])
    test_UE = np.array(UE_database[900:1000])
    test_IAB = np.array(IAB_database[900:1000])

    print(UE_database)
    print(IAB_database)



    batch_v = [50]
    learn_v = [1e-3]
    reguCost_v = [1e-3]
    # Best at b = 100 | l = 0.0001 | r = 0.001:
    # Best of S02_model_V2
    # [Epoch]: 131, [Train Loss]: 1.584E-08 , [Train Capacity Loss]: 2.782264 Mbps | [Valid Loss] 3.784E-07 , [Valid Capacity Loss]: 0.904608 Mbps
    # [Epoch]: 126, [Train Loss]: 2.628E-08 , [Train Capacity Loss]: 2.886908 Mbps | [Valid Loss] 1.337E-06 , [Valid Capacity Loss]: 1.840547 Mbps
    # [Epoch]: 108, [Train Loss]: 5.687E-08 , [Train Capacity Loss]: 2.781952 Mbps | [Valid Loss] 1.968E-06 , [Valid Capacity Loss]: 1.937512 Mbps
    # [Epoch]: 84, [Train Loss]: 6.013E-08 , [Train Capacity Loss]: 1.756789 Mbps | [Valid Loss] 1.035E-06 , [Valid Capacity Loss]: 2.650486 Mbps
    # [Epoch]: 64, [Train Loss]: 4.350E-07 , [Train Capacity Loss]: 1.949657 Mbps | [Valid Loss] 4.407E-07 , [Valid Capacity Loss]: 1.701459 Mbps

    # 'S02_model_V3', ResourceAllocationDynamicGelu,  b = 50 | l = 1e-3 | r = 1e-3
    # [Epoch]: 45, [Train Loss]: 4.371E-07 , [Train Capacity Loss]: 2.214261 Mbps | [Valid Loss] 1.649E-06 , [Valid Capacity Loss]: 2.741883 Mbps

    # WIN: 'S02_model_V4', ResourceAllocationDynamicGelu,  b = 50 | l = 1e-3 -> 1e-4 [60]  | r = 1e-3
    # [Epoch]: 106, [Train Loss]: 2.483E-08 , [Train Capacity Loss]: 2.210267 Mbps | [Valid Loss] 1.558E-06 , [Valid Capacity Loss]: 2.447651 Mbps

    # 'S02_model_V5', 46, resourceAllocation_Dynamic1, b = 50 | l = 1e-3 | r = 0
    for b in batch_v:
        for l in learn_v:
            for r in reguCost_v:
                print('minibatch_size = {} | learning_rate = {} | RegulationCost = {}'.format(b, l, r))
                train_model(TrainData_IAB=train_IAB,
                            TrainData_UE=train_UE,
                            ValidData_IAB=valid_IAB,
                            ValidData_UE=valid_UE,
                            model=nnmod.ResourceAllocationDynamicGelu(),
                            minibatch_size=b,
                            epoch=10,
                            learning_rate=l,
                            wd=0,
                            save_model=False,
                            directory='S02_model_V3',
                            lr_change=True,
                            RegulationCost=r)


if __name__ == '__main__':
    main()
