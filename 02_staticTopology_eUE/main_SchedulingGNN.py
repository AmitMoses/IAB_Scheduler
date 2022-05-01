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
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    isGenerate = False
    # main_path = os.path.dirname(os.path.abspath(__file__))
    main_path = '/common_space_docker/IAB_Scheduler'
    raw_paths_IAB_graph = main_path + '/GraphDataset/data/raw/'
    processed_dir_IAB_graph = main_path + '/GraphDataset/data/processed/'

    print(main_path)
    if isGenerate:
        IAB_Gdatabase = data_gen.process(raw_paths_IAB_graph, processed_dir_IAB_graph)
    else:
        IAB_Gdatabase = data_gen.load(processed_dir_IAB_graph)

    path_IAB = main_path + '/database/DynamicTopology/e6_m20_d3/IAB_database.csv'
    path_UE = main_path + '/database/DynamicTopology/e6_m20_d3/UE_database.csv'

    IAB_database = pd.read_csv(path_IAB)
    UE_database = pd.read_csv(path_UE)

    print(IAB_database.loc[0])
    print(IAB_Gdatabase[0].x)


if __name__ == '__main__':
    main()