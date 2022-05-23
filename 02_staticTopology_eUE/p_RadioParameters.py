__author__ = 'Amit'
# Radio parameters:

# General
eps = 1e-15
Total_BW = 550e6
CQI2efficiency = {
    0: 0,
    1: 0.1523,
    2: 0.3770,
    3: 0.8770,
    4: 1.4766,
    5: 1.9141,
    6: 2.4063,
    7: 2.7305,
    8: 3.3223,
    9: 3.9023,
    10: 4.5234,
    11: 5.1152,
    12: 5.5547,
    13: 6.2266,
    14: 6.9141,
    15: 7.4063
}

# Features
feature_num = 6
feature_num_old = 4
train_feature = 6

# IAB
IAB_num = 10
IAB_node_num = IAB_num - 1
backhaul_num = 1
maxUEperBS = 20

# UE
UE_num = 100
access_num = 2
