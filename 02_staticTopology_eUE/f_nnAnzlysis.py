__author__ = 'Amit'

import torch
import numpy as np
import matplotlib.pyplot as plt
import p_RadioParameters as rp
import f_SchedulingDataProcess as datap
import main_SchedulingProject as cost

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(mdl, folder, epoch):
    main_path = '/common_space_docker/IAB_scheduler/saved_model/'
    main_path = '../saved_model/'
    pt = main_path + str(folder) + '/epoch-' + str(epoch) + '.pt'
    mdl.load_state_dict(torch.load(pt, map_location=torch.device(device)))
    mdl.to(device)
    return mdl


def allocation_ability(cap, capCost):
    scr = 100 - np.mean(capCost/cap)*100
    return np.round(scr, 6)


# Plots for analysis functions:
# Resource allocation per IAB, requested vs allocation
def allocat_per_iab_boxplot(requested, allocation):
    data_list =[]
    for iab_num in range(0, 10):
        data_list.append(requested[:, iab_num, 0])
        data_list.append(allocation[:, iab_num, 0])
        data_list.append(requested[:, iab_num, 1])
        data_list.append(allocation[:, iab_num, 1])
    # configure and plot
    fig = plt.figure(figsize=(25, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_list, patch_artist=True)

    # colors
    colors = []
    for i in range(0, 20):
        colors.append('red')
        colors.append('blue')

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # ticks
    lbls = []
    for iab_num in range(1, 10):
        # for labels
        lbls.append('IAB' + str(iab_num) + ':DL')
        lbls.append('')
        lbls.append('IAB' + str(iab_num) + ':UL')
        lbls.append('')
    lbls.append('IAB' + str('10') + ':DL')
    lbls.append('')
    lbls.append('IAB' + str('10') + ':UL')

    ax.set_yticks(np.arange(0, 100, 5))  # consider to use max() and min()
    ax.set_xticks(np.arange(1.5, 40, 1))
    ax.set_xticklabels(lbls)

    # plot details
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ['Requested', 'Allocation'], loc='upper right')
    ax.grid()
    ax.set_title('Resource allocation per IAB')
    plt.show()


# Capacity (UL+DL): Requested vs Allocation
def cap_req_vs_alloc_bars(capacity, efficiency, test_pred):
    # Data process
    # CapacityCost = cost.capacity_cost(test_pred, Data_UEbatch, Data_IABbatch)

    CapacityCost = capacity - test_pred * rp.Total_BW * efficiency / 1e6
    index = torch.where(CapacityCost < 0)
    CapacityCost[index] = 0
    # CapacityCost = capacity - CapacityCost

    capCost = torch.sum(CapacityCost, dim=(1, 2)).detach().numpy()
    cap = torch.sum(capacity, dim=(1, 2)).detach().numpy()

    # Plot
    labels = list(range(1, cap.size+1))
    x = np.arange(len(labels))  # the label locations
    fig = plt.figure(figsize =(25, 7))
    ax = fig.add_subplot(111)
    ax.bar(x, capCost, 1, label='Capacity difference')
    ax.plot(x, np.ones_like(x)*np.mean(capCost), 'r--',
            label='Average difference')
    # plot details
    ax.set_xlim(0, cap.size+1)
    ax.set_ylabel('Capacity [Mbps]')
    ax.set_xlabel('Sample')
    ax.set_title('Capacity difference of Requested and Allocation (UL+DL)')
    ax.legend(prop={'size': 25}, loc='upper right')
    # ticks
    labels_t = list(range(0, cap.size, 50))
    x_t = np.array(labels_t)
    ax.set_xticks(x_t)
    ax.set_xticklabels(labels_t)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)
    plt.show()
    return cap, capCost


# Average unfulfilled links
def unfulfilled_links_bars(UE_capacity, UE_efficiency, UE_pred):
    out_UE = UE_pred * UE_efficiency * rp.Total_BW / 1e6

    req_vec = torch.reshape(UE_capacity, (-1,)).detach().numpy()  # requested data
    rel_vec = torch.reshape(out_UE, (-1,)).detach().numpy()  # allocation data

    links_per_samp = rp.maxUEperBS * rp.IAB_num * rp.access_num
    y_list = []
    for idx in range(0, req_vec.size, links_per_samp):
        req = req_vec[idx:idx + links_per_samp]
        rel = rel_vec[idx:idx + links_per_samp]

        # rel_index = np.where(rel == 0)  # find index where rel == 0
        # rel[rel_index] = rp.eps  # replies rel == 0 in eps. Prevents division by zero
        # rat = req / rel

        # rat[rat <= 1.0] = 1  # 1 means met the requirements
        # rat[rat > 1.0] = 0  # 0 means did not met the requirements
        rat = rel - req
        rat[rat >= 0] = 1
        rat[rat < 0] = 0
        y = np.sum(rat)  # /rat.size
        y_list.append(y)
    res = links_per_samp - np.asarray(y_list)
    res_mean = np.mean(res)
    # plot
    x = np.arange(1000)  # the label locations

    fig = plt.figure(figsize=(25, 7))
    ax = fig.add_subplot(111)

    # plot
    ax.bar(x, res, label='Unfulfilled links')
    ax.plot(x, np.ones_like(x) * res_mean, 'r--', label='Average')

    # plot details
    ax.set_xlim(0, x.size + 1)
    ax.set_ylabel('Links')
    ax.set_xlabel('Sample')

    ax.set_title('Number of lacking link connections')
    ax.legend(prop={'size': 25}, loc='upper right')
    ax.grid()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)
    plt.show()
    return res


# Simple resource allocation
def simple_resource_allocation(test_UE, test_IAB, iab_div):
    minibatch_size = test_UE.shape[0]
    # minibatch_size = 1
    Data_UEbatch, Data_IABbatch, Test_UEidx = datap.get_batch(test_UE, test_IAB, 0, minibatch_size)
    _, UE_capacity = datap.input_extract_for_cost(Data_UEbatch)
    _, IAB_capacity = datap.input_extract_for_cost(Data_IABbatch)
    # IAB_capacity[:, -1, :] = 0
    # allocation for UE links
    UE_norm = torch.sum(UE_capacity, dim=2)
    UE_norm = UE_norm.view((minibatch_size, -1, 1))
    IAB_norm = torch.sum(IAB_capacity, dim=2)
    IAB_norm = IAB_norm.view((minibatch_size, -1, 1))

    # need to replace nan to zero in IAB_norm
    access_pred = (UE_capacity / UE_norm) * (1 - iab_div)
    backhaul_pred = (IAB_capacity / IAB_norm) * iab_div

    gNB_pred = torch.sum(IAB_capacity, dim=2)
    # print(gNB_pred.shape)


    gNB_pred = gNB_pred.view(1000, 10, 1)
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

