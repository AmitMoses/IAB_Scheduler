__author__ = 'Amit'

import torch
import numpy as np
import matplotlib.pyplot as plt
import p_RadioParameters as rp
import f_SchedulingDataProcess as datap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(mdl, folder, epoch):
    main_path = '../saved_model/'
    pt = main_path + str(folder) + '/epoch-' + str(epoch) + '.pt'
    mdl.load_state_dict(torch.load(pt, map_location=torch.device(device)))
    mdl.to(device)
    return mdl


# fair_access_n_backhaul
def fair_access_n_backhaul(test_UE, test_IAB, iab_div):
    minibatch_size = test_UE.shape[0]
    Data_UEbatch, Data_IABbatch, Test_UEidx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    UE_eff, UE_capacity = datap.input_extract_for_cost(Data_UEbatch)
    IAB_eff, IAB_capacity = datap.input_extract_for_cost(Data_IABbatch)
    UE_bw = UE_capacity/UE_eff
    IAB_bw = IAB_capacity/IAB_eff
    IAB_capacity[:, -1, :] = 0
    # allocation for UE links
    UE_norm = torch.sum(UE_bw, dim=2)
    UE_norm = UE_norm.view((minibatch_size, -1, 1))
    IAB_norm = torch.sum(IAB_bw, dim=2)
    IAB_norm = IAB_norm.view((minibatch_size, -1, 1))

    # need to replace nan to zero in IAB_norm
    access_pred = (UE_bw / UE_norm) * (1 - iab_div)
    backhaul_pred = (IAB_bw / IAB_norm) * iab_div

    gNB_pred = torch.sum(IAB_bw, dim=2)
    # print(gNB_pred.shape)


    gNB_pred = gNB_pred.view(minibatch_size, 10, 1)
    # print(gNB_pred.shape)

    gNB_pred = gNB_pred
    # print(gNB_pred.shape)
    gNB_pred_norm = torch.sum(gNB_pred, dim=1)
    # print(gNB_pred_norm.shape)
    # print(gNB_pred_norm)
    gNB_pred_norm = gNB_pred_norm.view((minibatch_size, -1, 1))

    gNB_pred = gNB_pred/gNB_pred_norm
    # print(gNB_pred.shape)
    # print(gNB_pred[0])
    # print(torch.sum(gNB_pred, dim=(1,2)))

    pred = torch.cat((access_pred, backhaul_pred), dim=2) * gNB_pred
    pred[torch.isnan(pred)] = 0

    # print(pred.shape)
    # print(torch.sum(pred, dim=(1,2)))
    return pred


# Equal Resource allocation
def equal_resource(test_UE, test_IAB):
    minibatch_size = test_UE.shape[0]

    Data_UEbatch, Data_IABbatch, Test_UEidx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    _, UE_capacity = datap.input_extract_for_cost(Data_UEbatch)
    _, IAB_capacity = datap.input_extract_for_cost(Data_IABbatch)
    IAB_capacity[:, -1, :] = 0

    link_indicator = torch.cat((UE_capacity, IAB_capacity), dim=2) / (torch.cat((UE_capacity, IAB_capacity), dim=2) + rp.eps)
    pred = link_indicator / torch.sum(link_indicator, dim=(1, 2)).view(-1, 1, 1)

    return pred


# Out-of-Band like allocation
def out_of_band_like(test_UE, test_IAB, split_factor=0.5):
    minibatch_size = test_UE.shape[0]
    backhaul_spectrum = split_factor
    access_spectrum = 1 - split_factor

    Data_UEbatch, Data_IABbatch, Test_UEidx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    _, UE_capacity = datap.input_extract_for_cost(Data_UEbatch)
    _, IAB_capacity = datap.input_extract_for_cost(Data_IABbatch)
    IAB_capacity[:, -1, :] = 0

    UE_link_indicator = UE_capacity / (UE_capacity + rp.eps)
    IAB_link_indicator = IAB_capacity / (IAB_capacity + rp.eps)

    UE_pred = UE_link_indicator / torch.sum(UE_link_indicator, dim=(1, 2)).view(-1, 1, 1)
    IAB_pred = IAB_link_indicator / torch.sum(IAB_link_indicator, dim=(1, 2)).view(-1, 1, 1)

    UE_pred = access_spectrum * UE_pred
    IAB_pred = backhaul_spectrum * IAB_pred

    pred = torch.cat((UE_pred, IAB_pred), dim=2)
    return pred


# Optimal
def optimal(test_UE, test_IAB):
    minibatch_size = test_UE.shape[0]

    Data_UEbatch, Data_IABbatch, Test_UEidx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    UE_eff, UE_capacity = datap.input_extract_for_cost(Data_UEbatch)
    IAB_eff, IAB_capacity = datap.input_extract_for_cost(Data_IABbatch)
    IAB_capacity[:, -1, :] = 0

    UE_bw = UE_capacity / UE_eff
    IAB_bw = IAB_capacity / IAB_eff
    link_bw = torch.cat((UE_bw, IAB_bw), dim=2)
    pred = link_bw / torch.sum(link_bw, dim=(1,2)).view(-1, 1, 1)

    return pred


def main():
    pass


if __name__ == '__main__':
    main()
