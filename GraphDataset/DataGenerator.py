
import pandas as pd
from glob import glob
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
import os
import sys
sys.path.insert(1, '../02_staticTopology_eUE/')
import p_RadioParameters as rp
import collections



def load(processed_dir):
    print('Loading...')
    dataset = []
    all_data_paths = glob(processed_dir + '*.pt')
    all_data_paths.sort()
    print()
    for index, datapoint in tqdm(enumerate(all_data_paths), total=len(all_data_paths)):
        temp = datapoint
        data = torch.load(datapoint)
        data.edge_index = data.edge_index.long()
        dataset.append(data)
    print('Loading complete')
    print(f'Data length: {len(dataset)}')
    return dataset


def process(raw_paths, processed_dir, test=False):
    # dataset = []
    print('Processing...')
    all_data_paths = glob(raw_paths + '*.csv')
    for index, datapoint in tqdm(enumerate(all_data_paths), total=len(all_data_paths)):
        datapoint_num = get_iteration_index(datapoint)
        # data_frame = pd.read_csv(datapoint)
        # data_frame = data_frame.fillna(0)
        data_frame = csv2frame(datapoint)

        # node features
        node_feats = _create_node_feat(data_frame)
        # edge index
        edge_index = _create_edge_index(data_frame[0:-1])
        # lable
        label = _create_dummy_label(node_feats)

        # dtype
        # node_feats = torch.tensor(node_feats, dtype=torch.float)
        # edge_index = torch.tensor(edge_index, dtype=torch.float)
        # label = torch.tensor(label, dtype=torch.float)

        # Create data object
        data = Data(x=node_feats,
                    edge_index=edge_index,
                    y=label,
                    )
        if test:
            torch.save(data,
                       os.path.join(processed_dir,
                                    f'data_test_{datapoint_num}.pt'))
        else:
            torch.save(data,
                       os.path.join(processed_dir,
                                    f'data_{datapoint_num}.pt'))
        # dataset.append(data)
    dataset = load(processed_dir)
    return dataset


def get_iteration_index(path):
    # index = str(int(path.split('/')[-1].split('.')[0][9:]) - 1)
    index = str(int(path.split('\\')[-1].split('.')[0][9:]) - 1)
    if len(index) == 1:
        pad_index = '000' + index
    elif len(index) == 2:
        pad_index = '00' + index
    elif len(index) == 3:
        pad_index = '0' + index
    else:
        pad_index = index
    # print(f'index: {index} | pad_index: {pad_index}')
    return pad_index


def csv2frame(path):
    frame = pd.read_csv(path)
    frame = frame.fillna(15)
    return frame


def _create_edge_index(df):
    src = df.EndNodes1 - 1
    dst = df.EndNodes2 - 1
    edge_index = torch.tensor([src, dst])
    # edge_index = torch.reshape(edge_index.T,(-1,2))
    edge_index = torch.reshape(edge_index, (2, -1))
    return edge_index


def _create_node_features_link(df):
    num_link_feat = 6
    num_uplink_feat = int(num_link_feat / 2)
    num_downlink_feat = int(num_link_feat / 2)
    num_IAB_nodes = 9

    edge_CQI = torch.tensor(df.CQI).view(1, -1)
    # print(edge_CQI.shape)
    edge_eff = torch.tensor([rp.CQI2efficiency[int(index)] for index in edge_CQI.T]).view(1, -1)
    edge_Capacity = torch.tensor(df.Capacity).view(1, -1)
    edge_bw = edge_Capacity/edge_eff

    edge_attr = torch.cat((edge_Capacity, edge_eff, edge_bw), dim=0)
    # edge_attr = torch.reshape(edge_attr,(-1,num_link_feat))
    # print(edge_attr)
    ul_idx, dl_idx = _find_ul_dl_index(df)
    # uplink_feat = torch.reshape(edge_attr[:, 0:num_IAB_nodes].T, (-1, num_uplink_feat))
    # downlink_feat = torch.reshape(edge_attr[:, num_IAB_nodes:].T, (-1, num_downlink_feat))
    uplink_feat = torch.reshape(edge_attr[:, ul_idx].T, (-1, num_uplink_feat))
    downlink_feat = torch.reshape(edge_attr[:, dl_idx].T, (-1, num_downlink_feat))
    # print(uplink_feat)
    # print(downlink_feat)
    node_feat_link = torch.concat((uplink_feat, downlink_feat), dim=1)

    # temp = df[['EndNodes1', 'EndNodes2']].values.tolist()
    return node_feat_link


def _find_ul_dl_index(df):
    link_list = df[['EndNodes1', 'EndNodes2']].values.tolist()
    bank = []
    ul_index = []
    dl_index = []
    for index, item in enumerate(link_list):
        sort_item = list(np.sort(item))
        if sort_item in bank:
            dl_index.append(index)
        else:
            ul_index.append(index)
            bank.append(sort_item)
    return ul_index, dl_index


def _create_node_features_type(df):
    type_node = 0
    type_donor = 1

    # nodeID = torch.tensor(np.unique(np.concatenate((np.array(df.EndNodes1), np.array(df.EndNodes2)))))
    linkID = int(len(df)/2 + 1)
    # nodeID = torch.reshape(nodeID,(1,-1))
    # x = type_node * np.ones(len(linkID))
    x = type_node * np.ones(linkID)
    x[-1] = type_donor
    x = np.reshape(x, (linkID, -1))
    return torch.tensor(x)


def _create_node_feat(df):
    node_feat_link = _create_node_features_link(df[0:-1])
    # TEMP: ADD IAB DONOR FEATURS LINK
    # pad with the last node's features
    # padd = torch.reshape(node_feat_link[-1], (1, -1))
    padd = torch.tensor(df.iloc[-1:].values, dtype=torch.float)
    padd = pad_creat(padd)
    node_feat_link = torch.concat((node_feat_link, padd), dim=0)
    # print(node_feat_link.shape)
    node_feat_type = _create_node_features_type(df[0:-1])
    # print(node_feat_type.shape)
    node_feat = torch.concat((node_feat_type, node_feat_link), dim=1)
    return node_feat.float()


def _create_dummy_label(node_feat_):
    uplink_eff = int(1)
    uplink_capacity = int(2)
    downlink_eff = int(3)
    downlink_capacity = int(4)
    # print(node_feat_[:,uplink_capacity])
    # print(node_feat_[:,uplink_cqi])
    # print(node_feat_[:,downlink_capacity])
    # print(node_feat_[:,downlink_cqi])
    eps = 1e-15
    dummy_label = node_feat_[:, uplink_capacity] / (node_feat_[:, uplink_eff] + eps) + \
                  node_feat_[:, downlink_capacity] / (node_feat_[:, downlink_eff] + eps)
    dummy_label_norm = dummy_label / (torch.sum(dummy_label) + eps)

    return dummy_label_norm.float()


def pad_creat(vec):
    """
    Genarate the IAB-Donor input vector for padding the table input dataset
    :param vec: [UL_capacity, UL_eff, DL_capacity, DL_eff]
    :return: padd_vec: [UL_capacity, UL_eff, UL_bw, DL_capacity, DL_eff, DL_bw]
    """
    # pad = torch.tensor(vec)
    pad = vec.clone().detach()
    insert_idx_1 = 2
    insert_idx_2 = 4+1
    UL_bw = pad[:, 0]/pad[:, 1]
    DL_bw = pad[:, 2]/pad[:, 3]
    pad = np.insert(pad, [insert_idx_1], UL_bw, axis=1)
    pad = np.insert(pad, [insert_idx_2], DL_bw, axis=1)
    # pad = np.append(pad, DL_bw, axis=1)
    return pad


def nan_check_process(dataset):
    count = 0
    for index, item in enumerate(dataset):
        nan_x = torch.sum(torch.isnan(item.x))
        nan_y = torch.sum(torch.isnan(item.y))
        if nan_x > 0 or nan_y > 0:
            print(f'index: {index}')
            print(item.y)
            # print(f'nan_x: {nan_x}')
            # print(f'nan_y: {nan_y}')
            count += 1
    print(f'number of nan contain samples: {count}')


def nan_check_raw(paths):
    all_data_paths = glob(paths + '*.csv')
    count = 0
    for index, path in enumerate(all_data_paths):
        item_frame = csv2frame(path)
        item_tensor = torch.tensor(item_frame.values)
        nan_item = torch.sum(torch.isnan(item_tensor))
        if nan_item > 0:
            print(f'index: {index}')
            print(path)
            print(f'nan_item: {item_tensor}')
            count += 1
    print(f'number of nan contain samples: {count}')


def main():
    main_path = os.path.dirname(os.path.abspath(__file__))
    # raw_paths = main_path + '/data_v4/raw/'
    raw_paths = main_path + '\\data_v5\\raw\\'
    all_data_paths = glob(raw_paths + '*.csv')
    processed_dir = main_path + '/data_v5/processed/'
    Gdataset = process(raw_paths, processed_dir)
    # Gdataset = load(processed_dir)
    print(Gdataset[0].x)
    print(Gdataset[0].y)
    print(torch.sum(Gdataset[14].y))
    # print(len(Gdataset))

    # nan_check_process(Gdataset)
    # nan_check_raw(raw_paths)


if __name__ == '__main__':
    main()
