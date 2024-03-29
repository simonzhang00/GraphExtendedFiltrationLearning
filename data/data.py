import torch
import numpy as np
import torch_geometric
import torch_geometric.data

from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from ogb.graphproppred import PygGraphPropPredDataset
from .synthetic import Synthetic
from .utils import my_collate

POWERFUL_GNN_DATASET_NAMES =  ["PTC_PGNN"]
TU_DORTMUND_DATASET_NAMES = [
    "NCI1", "PTC_MR", 'PTC_FM', 
    'PTC_FR', 'PTC_MM', "PROTEINS", 
    "REDDIT-BINARY", "REDDIT-MULTI-5K", 
    "ENZYMES", "DD", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "COLLAB"
]
OGB_DATASET_NAMES= ['ogbg-molhiv', 'ogbg-ppa', 'ogbg-molpcba', 'ogbg-code2',
                            'ogbg-molbace',
                            'ogbg-molbbbp',
                            'ogbg-molclintox',
                            'ogbg-molmuv',
                            'ogbg-molsider',
                            'ogbg-moltox21',
                            'ogbg-moltoxcast']
SYNTHETIC_DATASET_NAMES= ['SimpleRings', 'PinWheels', 'Simple2Cycles']
BENCHMARK_GNN_DATASET_NAMES= ['MNIST','CIFAR10']

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        assert isinstance(X, list)
        self.data = X
        self.y= y
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for i in range(len(self.data)):
            yield self.data[i]
    
    
class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        assert isinstance(indices, (list, tuple))
        self.ds = dataset
        self.indices = tuple(indices)
        self.y= dataset.y[indices]
        assert len(indices) <= len(dataset)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]

def load_powerfull_gnn_dataset_PTC():
    has_node_features = False
    
    dataset_name = "PTC"
    path = "/home/pma/chofer/repositories/powerful-gnns/dataset/{}/{}.txt".format(dataset_name, dataset_name)

    with open(path, 'r') as f:
        num_graphs = int(f.readline().strip())

        data = []

        graph_label_map = {}
        node_lab_map = {}

        for i in range(num_graphs):
            row = f.readline().strip().split()
            num_nodes, graph_label = [int(w) for w in row]

            if graph_label not in graph_label_map:
                graph_label_map[graph_label] = len(graph_label_map)

            graph_label = graph_label_map[graph_label]


            nodes = []
            node_labs = []
            edges = []
            node_features = []

            for node_id in range(num_nodes):
                nodes.append(node_id)

                row = f.readline().strip().split()

                node_lab = int(row[0])

                if node_lab not in node_lab_map:
                    node_lab_map[node_lab] = len(node_lab_map)

                node_labs.append(node_lab_map[node_lab])

                num_neighbors = int(row[1])            
                neighbors = [int(i) for i in row[2:num_neighbors+2]]
                assert num_neighbors == len(neighbors)

                edges += [(node_id, neighbor_id) for neighbor_id in neighbors]

                if has_node_features:                
                    node_features = [float(i) for i in row[(2 + num_neighbors):]]
                    assert len(node_features) != 0           

            x = torch.tensor(node_labs, dtype=torch.long)

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_index = edge_index.permute(1, 0)
            tmp = edge_index.index_select(0, torch.tensor([1, 0]))
            edge_index = torch.cat([edge_index, tmp], dim=1)

            y = torch.tensor([graph_label])
            
            d = torch_geometric.data.Data(
                    x=x, 
                    edge_index=edge_index, 
                    y = y
                )

            d.num_nodes = num_nodes
            
            data.append(d)      

    max_node_lab = max([d.x.max().item() for d in data]) + 1
    eye = torch.eye(max_node_lab, dtype=torch.long)
    for d in data:
        node_lab = eye.index_select(0, d.x)
        d.x = node_lab

    ds = SimpleDataset(data)
    ds.name = dataset_name

    return ds


def get_boundary_info(g):
    
    e = g.edge_index.permute(1, 0).sort(1)[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long)

def enhance_SYNTHETICDataset(ds):
    X = []
    targets = []

    max_degree_by_graph = []
    num_nodes = []
    num_edges = []

    for d in ds:
        targets.append(d.y)

        boundary_info = get_boundary_info(d)
        d.boundary_info = boundary_info

        num_nodes.append(d.num_nodes)
        num_edges.append(boundary_info.size(0))

        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(d.boundary_info.flatten().tolist()).items():
            degree[k] = v
        max_degree_by_graph.append(degree.max().item())

        d.node_deg = degree
        X.append(d)

    max_node_deg = max(max_degree_by_graph)

    num_node_lab = None
    if hasattr(X[0], 'x') and X[0].x is not None and False:
        print("USING NODE LABEL")
        all_node_lab = []
        for d in X:
            assert d.x.sum() == d.x.size(0)  # really one hot encoded?
            node_lab = d.x.argmax(1).tolist()
            d.node_lab = node_lab
            all_node_lab += node_lab

        all_node_lab = set(all_node_lab)
        num_node_lab = len(all_node_lab)
        label_map = {k: i for i, k in enumerate(sorted(all_node_lab))}

        for d in X:  # numbers from consecutive indices represent the one hot encodings now
            d.node_lab = [label_map[f] for f in d.node_lab]
            d.node_lab = torch.tensor(d.node_lab, dtype=torch.long)
    else:
        for d in X:
            d.node_lab = None

    new_ds = SimpleDataset(X, ds.data.y)

    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    print(targets)
    new_ds.num_classes = min(len(set(targets)), len(set([t.item() for t in  targets])))
    new_ds.num_node_lab = num_node_lab

    return new_ds

def enhance_PYGYDataset(ds):
    X = []
    targets = []

    max_degree_by_graph = []
    num_nodes = []
    num_edges = []


    for d in ds:
        targets.append(d.y.item())


        boundary_info = get_boundary_info(d)
        d.boundary_info = boundary_info

        num_nodes.append(d.num_nodes)
        num_edges.append(boundary_info.size(0))

        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(d.boundary_info.flatten().tolist()).items():
            degree[k] = v
        max_degree_by_graph.append(degree.max().item())

        d.node_deg = degree
        X.append(d)

    max_node_deg = max(max_degree_by_graph)


    for d,dd in zip(X,ds):
        if dd.x is not None:
            d.node_lab= torch.tensor(dd.x.tolist())
        else:
            d.node_lab= torch.tensor([])
    if X[0].x is not None:
        num_node_lab= (X[0].x.shape[1])
    else:
        num_node_lab= 0

    new_ds = SimpleDataset(X, ds.data.y)

    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes = len(set(targets))
    new_ds.num_node_lab = num_node_lab

    return new_ds

def enhance_TUDataset(ds):
      
    X = []
    targets = []
    
    max_degree_by_graph = []
    num_nodes = []
    num_edges = []
    
    for d in ds:
        targets.append(d.y.item())

        boundary_info = get_boundary_info(d)
        d.boundary_info = boundary_info

        num_nodes.append(d.num_nodes)
        num_edges.append(boundary_info.size(0))
        
        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(d.boundary_info.flatten().tolist()).items():
            degree[k] = v
        max_degree_by_graph.append(degree.max().item())
            
        d.node_deg = degree
        X.append(d)
        
    max_node_deg =  max(max_degree_by_graph)
    
    num_node_lab = None

    if hasattr(X[0], 'x') and X[0].x is not None and False:
        if X[0].x.sum() == X[0].x.size(0):
            all_node_lab = []
            for d in X:
                assert d.x.sum() == d.x.size(0) # really one hot encoded?
                node_lab = d.x.argmax(1).tolist()
                d.node_lab = node_lab
                all_node_lab += node_lab
            all_node_lab = set(all_node_lab)
            num_node_lab = len(all_node_lab)
            label_map = {k: i for i, k in enumerate(sorted(all_node_lab))}

            for d in X:# numbers from consecutive indices represent the one hot encodings now
                d.node_lab = [label_map[f] for f in d.node_lab]
                d.node_lab = torch.tensor(d.node_lab, dtype=torch.long)
        else:
            for d in X:
                d.node_lab= d
    else:
        for d in X:
            d.node_lab = None

    new_ds =  SimpleDataset(X, ds.data.y)
    
    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes   = len(set(targets))
    new_ds.num_node_lab = num_node_lab
    
    return new_ds


def enhance_BenchmarkDataset(ds):
    X = []
    targets = []

    max_degree_by_graph = []
    num_nodes = []
    num_edges = []

    for d in ds:
        dd= torch_geometric.data.Data(x= d.x, edge_index=d.edge_index, edge_attr= d.edge_attr, y= d.y)
        targets.append(d.y.item())

        boundary_info = get_boundary_info(d)
        dd.boundary_info = boundary_info

        num_nodes.append(d.num_nodes)
        num_edges.append(boundary_info.size(0))

        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(dd.boundary_info.flatten().tolist()).items():
            degree[k] = v
        max_degree_by_graph.append(degree.max().item())

        dd.node_deg = degree
        X.append(dd)

    max_node_deg = max(max_degree_by_graph)

    num_node_lab = None

    if hasattr(X[0], 'x') and X[0].x is not None and False:
        if X[0].x.sum() == X[0].x.size(0):
            all_node_lab = []
            for d in X:
                assert d.x.sum() == d.x.size(0)  # really one hot encoded?
                node_lab = d.x.argmax(1).tolist()
                d.node_lab = node_lab
                all_node_lab += node_lab
            all_node_lab = set(all_node_lab)
            num_node_lab = len(all_node_lab)
            label_map = {k: i for i, k in enumerate(sorted(all_node_lab))}

            for d in X:  # numbers from consecutive indices represent the one hot encodings now
                d.node_lab = [label_map[f] for f in d.node_lab]
                d.node_lab = torch.tensor(d.node_lab, dtype=torch.long)
        else:
            for d in X:
                d.node_lab = d
    else:
        for d in X:
            d.node_lab = None

    new_ds = SimpleDataset(X, ds.data.y)

    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes = len(set(targets))
    new_ds.num_node_lab = num_node_lab

    return new_ds


def dataset_factory(dataset_name, split= "cv", verbose=True):
    if dataset_name in TU_DORTMUND_DATASET_NAMES:

        path = 'data/{}/'.format(dataset_name)
        dataset = TUDataset(path, name=dataset_name)

    elif dataset_name in POWERFUL_GNN_DATASET_NAMES:
        if dataset_name == "PTC_PGNN":
            dataset = load_powerfull_gnn_dataset_PTC()
    elif dataset_name in OGB_DATASET_NAMES:
        dataset= PygGraphPropPredDataset(name = dataset_name)
    elif dataset_name in BENCHMARK_GNN_DATASET_NAMES:
        path = 'data/{}/'.format(dataset_name)
        dataset= GNNBenchmarkDataset(root= path, name= dataset_name, split= split)
    elif dataset_name in SYNTHETIC_DATASET_NAMES:
        if dataset_name=='SimpleRings':
            dataset= Synthetic.SimpleRings(root=".")
        elif dataset_name=='PinWheels':
            dataset= Synthetic.PinWheels(root=".")
        elif dataset_name=='Simple2Cycles':
            dataset= Synthetic.SimpleTwoCycles(root='.')
    else:
        raise ValueError("dataset_name not in {}".format(TU_DORTMUND_DATASET_NAMES + POWERFUL_GNN_DATASET_NAMES+OGB_DATASET_NAMES+SYNTHETIC_DATASET_NAMES))
    ds_name = dataset.name
    if dataset_name in OGB_DATASET_NAMES:
        dataset= enhance_PYGYDataset(dataset)
    elif dataset_name in SYNTHETIC_DATASET_NAMES:
        dataset= enhance_SYNTHETICDataset(dataset)
    elif dataset_name in BENCHMARK_GNN_DATASET_NAMES:
        dataset= enhance_BenchmarkDataset(dataset)
    else:
        dataset = enhance_TUDataset(dataset)

    if verbose:
        print("# Dataset: ", ds_name)    
        print('# num samples: ', len(dataset))
        print('# num classes: ', dataset.num_classes)
        print('#')
        print('# max node degree: ', dataset.max_node_deg)
        print('# num node label ', dataset.num_node_lab)
        print('#')
        print('# avg number of nodes: ', dataset.avg_num_nodes)
        print('# avg number of edges: ', dataset.avg_num_edges)

    
    return dataset

def OGB_dataset_factory(dataset_name, verbose= True):
    if dataset_name in OGB_DATASET_NAMES:
        dataset = PygGraphPropPredDataset(name=dataset_name)
        split_idx= dataset.get_idx_split()
        task_type= dataset.task_type
        eval_metric= dataset.eval_metric
    else:
        raise ValueError("dataset_name not in {}".format(
             OGB_DATASET_NAMES ))
    ds_name = dataset.name
    if dataset_name in OGB_DATASET_NAMES:
        dataset = enhance_PYGYDataset(dataset)

    if verbose:
        print("# Dataset: ", ds_name)
        print('# num samples: ', len(dataset))
        print('# num classes: ', dataset.num_classes)
        print('#')
        print('# max node degree: ', dataset.max_node_deg)
        print('# num node label ', dataset.num_node_lab)
        print('#')
        print('# avg number of nodes: ', dataset.avg_num_nodes)
        print('# avg number of edges: ', dataset.avg_num_edges)

    return dataset, split_idx, task_type, eval_metric

def train_test_val_split(
    dataset,
    seed=0, 
    n_splits=10,
    verbose=True, 
    validation_ratio=0.0):

    skf = StratifiedKFold(
        n_splits=n_splits, 
        shuffle = True,#True,
        random_state = seed, 
    )

    targets = [x.y.item() for x in dataset]
    split_idx = list(skf.split(np.zeros(len(dataset)), targets))

    if verbose:
        print('# num splits: ', len(split_idx))
        print('# validation ratio: ', validation_ratio)

    split_ds = []
    split_i = []
    for train_i, test_i in split_idx:
        not_test_i, test_i = train_i.tolist(), test_i.tolist()

        if validation_ratio == 0.0:
            validation_i = []
            train_i = not_test_i

        else:
            skf = StratifiedShuffleSplit(
                n_splits=1, 
                random_state = seed, 
                test_size=validation_ratio
            )

            targets = [dataset[i].y.item() for i in not_test_i]
            train_i, validation_i = list(skf.split(np.zeros(len(not_test_i)), targets))[0]
            train_i, validation_i = train_i.tolist(), validation_i.tolist()  

            # We need the indices w.r.t. the original dataset 
            # not w.r.t. the current train fold ... 
            train_i = [not_test_i[j] for j in train_i]
            validation_i = [not_test_i[j] for j in validation_i]
            
        assert len(set(train_i).intersection(set(validation_i))) == 0
        
        train = Subset(dataset, train_i)
        test = Subset(dataset, test_i)
        validation = Subset(dataset, validation_i)

        assert sum([len(train), len(test), len(validation)]) == len(dataset)

        split_ds.append((train, test, validation))
        split_i.append((train_i, test_i, validation_i))

    return split_ds, split_i
