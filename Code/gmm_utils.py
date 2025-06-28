import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import draw_graph, get_class_label
from geomix_utils import lgw, proj_graph
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import sys
from scipy.stats import bernoulli



def prepare_data(dataset,model, args):
    set_labels = set()
    model.eval()
    ## randomly select samples and mixup by geomix
    data_out = []    
    for ele in tqdm(range(len(dataset))):
        for _ in range(args.aug_per_graph) : 
            edge_index_ele = dataset[ele].edge_index.to(args.device)
            x_ele = dataset[ele].x.to(args.device)
            if 'edge_weights' in dataset[ele]:
                edge_weights_ele = dataset[ele].edge_weights.float()
            else:
                edge_weights_ele = None
            if args.model == 'GCN' :
                x = F.relu(model.conv1(x_ele, edge_index_ele, edge_weight = edge_weights_ele)).detach()
                for conv in model.convs:
                    x = F.relu(conv(x, edge_index_ele, edge_weight = edge_weights_ele)).detach()
                x =  torch.mean(x, 0)
            if args.model == 'GIN' :
                x = F.relu(model.lin1(x_ele))
                x = F.relu(model.conv1(x, edge_index = edge_index_ele, edge_weight = edge_weights_ele))
                x = model.bn1(x)
                x =  torch.mean(x, 0)
            x = x.detach().unsqueeze(0)# .cpu().numpy()
            data_out.append(Data(x = x, y = dataset[ele].y))
            set_labels.add(torch.argmax(dataset[ele].y).item())
    return data_out, set_labels





def GMMs_aug(splt_train_dataset,args):
    ## randomly select samples and mixup by geomix
    print('GMM Models ...')
    data_out = []
    train_set_labels = set(splt_train_dataset.keys())
    for l_ in train_set_labels:

        train_dataset_l_ = splt_train_dataset[l_]
        data_out = data_out  + train_dataset_l_
        new_x = np.array([])
        for t_, ele in tqdm(enumerate(np.arange(len(train_dataset_l_)))):
            x_ele = train_dataset_l_[ele].x.cpu().numpy()
            if t_ ==0 : 
                new_x = x_ele
            else: 
                new_x = np.concatenate([new_x,x_ele], 0 )
        gm_ele = GaussianMixture(n_components=args.gmm_components).fit(new_x)
        num_aug = int(args.aug_per_graph * new_x.shape[0])
        
        for _ in range(num_aug) :
            try: 
                x_gen= gm_ele.sample()[0]
                data_out.append(Data(x = torch.tensor(x_gen).to(args.device).float(), y = train_dataset_l_[ele].y))
                #print(train_dataset_l_[ele].y, torch.tensor(x_gen).to(args.device).float().size(), train_dataset_l_[ele].x.size())
            except :
                WWW = 1
                print(555555555555555)

    return data_out
