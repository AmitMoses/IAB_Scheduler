import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import os
import DataGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../02_staticTopology_eUE/')
import f_SchedulingDataProcess_2 as datap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GCN model with 2 layers
class Net(torch.nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 1)
        self.fc1 = nn.Linear(10, 10)
        self.out = nn.Softmax(dim=1)

    def forward(self, data, batch_size):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        # Layer 3
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        # Output Layer
        x = self.out(x)
        return x


def train(model, train_data, model_config):
    train_loss = []
    valid_loss = []
    optimizer_name = "Adam"
    # optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['wd'])
    epochs = 200
    train_loader = DataLoader(train_data, batch_size=model_config['batch_size'])

    model.train()
    optimizer.zero_grad()

    for t in range(model_config['epochs']):
        for index, batch in enumerate(train_loader):
            # print(batch.x.shape)
            batch = batch.to(device)
            pred = model(batch, model_config['batch_size'])
            y_label = batch.y.view(model_config['batch_size'], -1)

            # a1 = torch.sum(torch.isnan(pred))
            # a2 = torch.sum(torch.isnan(y_label))
            # print(a1)
            # print(a2)

            # temp = F.mse_loss(pred, y_label)
            # print(F.mse_loss(pred, y_label))
            loss = F.mse_loss(pred, y_label)
            # loss =datap.topology_cost(pred, y_label)
            loss.backward()
            optimizer.step()
            # print(batch[0].x)
        train_loss.append(loss.item())
        print(f'epoch: {t} | Train loss = {loss.item()}')

    # plot
    plt.figure()
    plt.title('GNN loss \n b = {} | lr = {} | wd = {}'
              .format(model_config['batch_size'], model_config['lr'], model_config['wd']))
    plt.semilogy(range(model_config['epochs']), train_loss, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def main():
    isGenerate = False
    main_path = os.path.dirname(os.path.abspath(__file__))
    raw_paths = main_path + '/data/raw/'
    processed_dir = main_path + '/data/processed/'

    if isGenerate:
        Gdataset = DataGenerator.process(raw_paths, processed_dir)
    else:
        Gdataset = DataGenerator.load(processed_dir)

    trainData = Gdataset[0:9000]
    validData = Gdataset[9000:10000]
    minibatch_size = 100
    num_features = Gdataset[0].x.shape[1]

    config = {
        'lr': 1e-3,
        'wd': 1e-8,
        'epochs': 5,
        'batch_size': 100
    }
    # b = 500 | l = 0.001 ,1e-4, 1e-5  | w =
    # train(model=Gmodel,
    #       train_data=Gdataset,
    #       model_config=config
    #       )

    batch_v = [1000, 500]
    lr_v = [1e-3, 1e-4, 1e-5, 1e-6]
    wd_v = [0, 1e-6, 1e-3]
    for b in batch_v:
        for lr in lr_v:
            for wd in wd_v:
                Gmodel = Net(num_features=num_features).to(device)
                config['batch_size'] = b
                config['lr'] = lr
                config['wd'] = wd
                print(f'minibatch_size = {b} | learning_rate = {lr} | weight_decay = {wd}')
                train(model=Gmodel,
                      train_data=trainData,
                      model_config=config
                      )


if __name__ == '__main__':
    main()
