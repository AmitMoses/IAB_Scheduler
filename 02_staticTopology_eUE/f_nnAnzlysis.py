__author__ = 'Amit'

import torch
import numpy as np
import matplotlib.pyplot as plt
import p_RadioParameters as rp
import f_SchedulingDataProcess as datap
import seaborn as sns
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def capacity_cost(capacity, test_pred, efficiency):
    CapacityCost = capacity - test_pred * rp.Total_BW * efficiency / 1e6
    index = torch.where(CapacityCost < 0)
    CapacityCost[index] = 0
    return CapacityCost


def allocation_ability(cap, capCost):
    scr = 100 - np.mean(capCost / cap) * 100
    return np.round(scr, 6)


# Plots for analysis functions:
# Resource allocation per IAB, requested vs allocation
def allocat_per_iab_boxplot(requested, allocation):
    data_list = []
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
def cap_req_vs_alloc_bars(capacity, efficiency, test_pred, is_plot=True):
    """
    :param capacity:
    :param efficiency:
    :param test_pred:
    :param is_plot: flag for plotting. If True print plot.
    :return:
    cap: average requested capacity
    capCost: average allocated capacity
    """
    # Data process
    CapacityCost = capacity_cost(capacity, test_pred, efficiency)

    capCost = torch.sum(CapacityCost, dim=(1, 2)).detach().numpy()
    cap = torch.sum(capacity, dim=(1, 2)).detach().numpy()

    # Plot
    if is_plot:
        labels = list(range(1, cap.size + 1))
        x = np.arange(len(labels))  # the label locations
        fig = plt.figure(figsize=(25, 7))
        ax = fig.add_subplot(111)
        ax.bar(x, capCost, 1, label='Capacity difference')
        ax.plot(x, np.ones_like(x) * np.mean(capCost), 'r--',
                label='Average difference')
        # plot details
        ax.set_xlim(0, cap.size + 1)
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
def unfulfilled_links_bars(UE_capacity, UE_efficiency, UE_pred, is_plot=True, is_access=True):
    sample_len = UE_pred.shape[0]
    out_UE = UE_pred * UE_efficiency * rp.Total_BW / 1e6

    req_vec = torch.reshape(UE_capacity, (-1,)).detach().numpy()  # requested data
    rel_vec = torch.reshape(out_UE, (-1,)).detach().numpy()  # allocation data
    if is_access:
        links_per_samp = rp.maxUEperBS * rp.IAB_num * rp.access_num
    else:
        links_per_samp = rp.IAB_num * rp.access_num

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
    if is_plot:
        x = np.arange(sample_len)  # the label locations

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


def bandwidth_usage(capacity_request, efficiency, pred):
    capacity_allocate = pred * rp.Total_BW * efficiency / 1e6
    capacity_usage = torch.min(capacity_request, capacity_allocate)
    bandwidth_total = capacity_usage / efficiency
    return torch.sum(bandwidth_total, dim=(1, 2))


def network_efficiency(capacity_request, efficiency, pred):
    capacity_allocate = pred * rp.Total_BW * efficiency / 1e6
    capacity_usage = torch.min(capacity_request, capacity_allocate)
    system_efficiency = capacity_usage / rp.Total_BW
    return torch.sum(system_efficiency, dim=(1, 2))


def capacity_loss_per_link(average_unfulfilled_links, average_difference):
    ad_np = np.array(average_difference)
    aul_np = np.array(average_unfulfilled_links)
    loss_per_link = list(ad_np / (aul_np + rp.eps))
    return loss_per_link


# Create bar plot
def draw_bar_plot(x, y, title, x_label, y_label, is_save=False):
    methods = x
    results = y

    # optimal_method = x[-1]
    # optimal_result = y[-1]

    plt.figure()
    bar_plot = sns.barplot(x=methods, y=results)
    ax = bar_plot.axes

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    # ax.axhline(optimal_result, ls='--')
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)
    ax.set_title(title)
    plt.grid()

    if is_save:
        save_dir = '../saved_fig/'
        plt.savefig(save_dir + title + '.png')

    plt.show()


def draw_box_plot(x, y, title, x_label, y_label):
    methods = x
    results = np.array(y).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(labels=methods, x=results)
    ax.set_title(title)
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)

    plt.grid()
    plt.show()


def seaborn_box_plot(x, y, title, x_label, y_label, is_save=False):
    methods = x
    results = np.array(y).T
    plt.figure()
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(methods)
    ax.set_title(title)
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)
    ax.grid()

    if is_save:
        save_dir = '../saved_fig/'
        plt.savefig(save_dir + title + '.png')

    plt.show()


# def draw_access_backhaul_plot(data_ue, data_iab, methods):
#     print('hi')
#     df = pd.DataFrame(
#         [
#             data_ue,
#             data_iab,
#
#         ], columns=methods)
#     df.index = ['ue', 'iab']
#     plt.figure()
#     sns.barplot(data=df.loc['ue'])
#     sns.barplot(x="ue", data=df)
#     plt.show()
#     pass


def draw_bar_plot_2(x, y1, y2, title, x_label, y_label):
    methods = x[0:-1]
    results1 = y1[0:-1]
    results2 = y2[0:-1]

    optimal_method = x[-1]
    optimal_result1 = y1[-1]
    optimal_result2 = y2[-1]

    plt.figure()
    bar_plot = sns.barplot(x=methods, y=results1)
    ax = bar_plot.axes
    ax.axhline(optimal_result1, ls='--')
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)
    ax.set_title(title)
    plt.grid()
    plt.show()
