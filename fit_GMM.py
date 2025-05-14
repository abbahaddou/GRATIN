import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch_geometric.nn import global_mean_pool
import torch
import numpy as np
import time
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from model import GCN, GIN_Net, APPNP_Net
from gmm_utils import GMMs_aug, prepare_data
from sklearn.model_selection import KFold
from utils import preprocess
import warnings
import torch.nn.functional as F
from torch.nn import Linear
import sys
warnings.filterwarnings("ignore")
def mixup_cross_entropy_loss(input, target, size_average=True):
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def train_pooling(model, optimizer, train_loader):
    model.train()
    
    train_loss = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(args.device)
        out = model(data)  # Perform a single forward pass.
        loss = mixup_cross_entropy_loss(out, data.y)
        train_loss.append(loss.detach().item())
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    
    return np.mean(train_loss)


def test_pooling(model, loader):
    model.eval()

    correct = 0
    n = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        gnd = data.y.argmax(dim=1)
        n = n + len(gnd)
        correct += int((pred == gnd).sum())  # Check against ground-truth labels.
    return correct / n  # Derive ratio of correct predictions.


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class Pooling_Net(torch.nn.Module):
    def __init__(self, num_hidden=32, num_classes=1):
        super(Pooling_Net, self).__init__()
        self.lin2 = Linear(num_hidden, num_classes)

    def reset_parameters(self):
        self.lin2.reset_parameters()

    def forward(self, data):
        x, batch = data.x,  data.batch
        x = F.dropout(x, args.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
parser = argparse.ArgumentParser()

## Training parameters
parser.add_argument("--data", type=str, default='MUTAG', choices= 'IMDB-BINARY|IMDB-MULTI|PROTEINS|MUTAG|PTC_MR|MSRC_9|FIRSTMM_DB|DD|REDDIT-BINARY|SYNTHETIC|NCI1')
parser.add_argument("--model", type=str, default='GCN', choices= 'GCN|GIN|APPNP')
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--aug_per_graph", type=int, default=1)
parser.add_argument("--gmm_components", type=int, default=10)

## Augmentation parameters
parser.add_argument("--dropout", type=float, default=0.8, help='dropout')
parser.add_argument("--aug_ratio", type=float, default=0.1, help='number of mixup pairs')
parser.add_argument("--num_graphs", type=int, default=10, help='number of mixup graphs per pair')
parser.add_argument("--num_nodes", type=int, default=20, help='number of nodes in the mixup graph')
parser.add_argument("--alpha_fgw", type=float, default=1.0, help='weight for GW term in FGW distance')
parser.add_argument("--sample_dist", type=str, default='uniform', choices='uniform|beta', help='mixup weight sample distribution')
parser.add_argument("--beta_alpha", type=float, default=5, help='Beta(alpha, beta)')
parser.add_argument("--beta_beta", type=float, default=0.5, help='Beta(alpha, beta)')
parser.add_argument("--uniform_min", type=float, default=0.0, help='Uniform(min,max)')
parser.add_argument("--uniform_max", type=float, default=5e-2, help='Uniform(min,max)')
parser.add_argument("--clip_eps", type=float, default=1e-3, help='threshold to filter out zero columns')

## other arguments
parser.add_argument('--cuda', type=int, default=0)

args = parser.parse_args()
if args.data == 'MUTAG':
    if args.model == 'GCN' : 
        args.dropout = 0.3
        args.gmm_components = 10

    elif args.model == 'GIN' : 
            args.dropout = 0.7
            args.gmm_components = 2
elif args.data == 'IMDB-BINARY':
    if args.model == 'GCN' : 
        args.dropout = 0.1
        args.gmm_components = 40

    elif args.model == 'GIN' : 
            args.dropout = 0.5
            args.gmm_components = 50
elif args.data == 'IMDB-MULTI':
    if args.model == 'GCN' : 
        args.dropout = 0.1
        args.gmm_components = 50

    elif args.model == 'GIN' : 
            args.dropout = 0.3
            args.gmm_components = 5


elif args.data == 'PROTEINS':
    if args.model == 'GCN' : 
        args.dropout = 0.3
        args.gmm_components = 10

    elif args.model == 'GIN' : 
            args.dropout = 0.1
            args.gmm_components = 30



elif args.data == 'DD':
    if args.model == 'GCN' : 
        args.dropout = 0.1
        args.gmm_components = 2

    elif args.model == 'GIN' : 
            args.dropout = 0.7
            args.gmm_components = 50
args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
random_state = 1234

def main(args):
    dataset = TUDataset(root='data/TUDataset', name=args.data)
    dataset = list(dataset)
    dataset, num_classes = preprocess(dataset)

    kf = KFold(n_splits=10, shuffle = True, random_state = random_state)
    acc = []
    
    train_time = []
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        
        train_index, val_index = np.split(train_index, [int(8 / 9 * len(train_index))])
        train_dataset = [dataset[j].to(args.device) for j in train_index]
        test_dataset = [dataset[j].to(args.device) for j in test_index]
        val_dataset = [dataset[j].to(args.device) for j in val_index]
        
        #train_dataset  = train_dataset[:200]
        if args.model == 'GCN':
            model = GCN(dataset[0].num_node_features,args.hidden_dim, num_classes).to(args.device)
        elif args.model == 'GIN':
            model = GIN_Net(dataset[0].num_node_features, args.hidden_dim, num_classes).to(args.device)
        elif args.model == 'APPNP':
            model = APPNP_Net(dataset[0].num_node_features, args.hidden_dim, num_classes).to(args.device)
        else:
            raise KeyError('Invalid model name!')
        model.load_state_dict(torch.load('./checkpoints/{}_{}_fold_{}.pth'.format(args.data, args.model, i))['model_state_dict'])
        model.eval()

        train_dataset, train_set_labels = prepare_data(train_dataset, model,args )
        test_dataset, test_set_labels = prepare_data(test_dataset, model,args )
        val_dataset, val_set_labels = prepare_data(val_dataset, model,args )

        splt_train_dataset = {}
        for l_ in train_set_labels: 
            splt_train_dataset[l_] = [d_ for d_ in train_dataset if torch.argmax(d_.y).item() == l_]

        t1 = time.time()
        print('Start Data augmentation ...')
        new_train_dataset = GMMs_aug(splt_train_dataset, args)
        print(f'Augmentation time: {time.time()-t1:3f}')
        ts = time.time()

        train_loader = DataLoader(new_train_dataset, batch_size=args.batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
        
        pooling_model = Pooling_Net(args.hidden_dim, num_classes).to(args.device)
        pooling_model.reset_parameters()
        optimizer = torch.optim.Adam(pooling_model.parameters(), lr=1e-2, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        
        early_stopping = EarlyStopping(tolerance=50, min_delta=0.1)
        for epoch in range(1, args.epochs+1):
            train_loss = train_pooling(pooling_model, optimizer, train_loader)
            train_acc = test_pooling(pooling_model, train_loader)
            scheduler.step()

            with torch.no_grad(): 
                val_acc = test_pooling(pooling_model, val_loader)
            early_stopping(train_acc, val_acc)
            if early_stopping.early_stop:
                test_acc = test_pooling(pooling_model, test_loader)
                print(f'Early breaking!')
                print(f'Fold: {i+1:01d}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                break
            if epoch % 50 == 0:
                test_acc = test_pooling(pooling_model, test_loader)
                print(f'Fold: {i+1:01d}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        train_time.append(time.time() - ts)
        test_acc = test_pooling(pooling_model, test_loader)
        print('test_acc ... ', test_acc)
        acc.append(test_acc)
    print('dataset: {}, model: {} avg_acc:{:.5f}, std:{:.5f}, time:{:.3f}, std:{:.3f}'.format(args.data,  args.model, np.mean(acc), np.std(acc), np.mean(train_time), np.std(train_time)))
    return np.mean(acc), np.std(acc), np.mean(train_time), np.std(train_time)
print(args.data, args.model)
mean_acc, std_acc, mean_train_time, std_train_time = main(args)
print('mean_acc  : ', mean_acc , std_acc)






