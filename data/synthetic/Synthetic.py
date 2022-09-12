import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data, InMemoryDataset
from random import randrange
import numpy as np

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class SimpleTwoCycles(InMemoryDataset):
    def __init__(self, root, num_graphs= 20, transform=None, pre_transform=None):
        self.num_graphs= num_graphs
        self.name= "SimpleTwoCycles"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        fix_seed(0)
    @property
    def raw_file_names(self):
       return ['twocycles_'+str(i)+'.pt' for i in range(self.num_graphs)]

    @property
    def processed_file_names(self):
        return ['twocycle_data.pt']

    #def download(self):
    #    # Download to `self.raw_dir`.
    #    path = download_url(url, self.raw_dir)
    #    ...

    def process(self):
        data_list= []
        idx = 0
        for n in range(int(self.num_graphs)):
            if n % 2==0:
                n= randrange(53,63)
                #for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                data = Data(x= torch.tensor([[1.,0.]]*n, dtype= torch.float),
                    edge_index=torch.tensor([[0,1],[1,2],[2,0]]+[[i,i+1] for i in range(3,n-1)]+[[n-1,3]]).transpose(0,1),y= torch.tensor([1]))

                data.y= torch.Tensor([1])
                print("data0; ", data)
                print("data0.y", data.y)
                print("data0.x: ", data.x)
            else:
                n = randrange(53, 63)
                k= int(n/2)
                data = Data(x= torch.tensor([[1.,0.]]*n, dtype= torch.float),
                            edge_index=torch.tensor([[i,i+1] for i in range(0,k-1)]+ [[k-1,0]]+ [[i,i+1] for i in range(k,n-1)]+ [[n-1,k]]).transpose(0, 1),
                            y=torch.tensor([0]))
                data.y = torch.Tensor([0])

                print(data)
            data.num_nodes = n
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            idx += 1
            data_list.append(data)
        print(data_list)
        data,slices= self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])

    def len(self):
        return self.num_graphs

class PinWheels(InMemoryDataset):
    def __init__(self, root, num_graphs= 100, transform=None, pre_transform=None):
        self.num_graphs= num_graphs
        self.name= "PinWheels"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        fix_seed(0)
    @property
    def raw_file_names(self):
       return ['pinwheels_'+str(i)+'.pt' for i in range(self.num_graphs)]

    @property
    def processed_file_names(self):
        return ['pinwheels_data.pt']

    #def download(self):
    #    # Download to `self.raw_dir`.
    #    path = download_url(url, self.raw_dir)
    #    ...

    def process(self):
        data_list= []
        idx = 0
        for n in range(int(self.num_graphs)):
            if n % 2==0:
                tendrils_per_node= randrange(10,13)
                n= tendrils_per_node*6+6
                #for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                data = Data(x= torch.tensor([[1,0]]*n, dtype= torch.float),
                    #x= torch.tensor([[1,0,0,0]]*3+[[0,1,0,0]]*3+[[0,0,1,0] if k in [0,1,2] else [0,0,0,1] for k in range(0,6) for i in range(1,tendrils_per_node+1) ]),
                            edge_index=torch.tensor([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3]]+[[k,6*i+k] for i in range(1,tendrils_per_node+1) for k in range(0,6)]).transpose(0,1),y= torch.tensor([1]))

                data.y= torch.Tensor([1])
            else:
                tendrils_per_node = randrange(10,13)
                n = tendrils_per_node * 6 + 6
                data = Data(x= torch.tensor([[1,0]]*n),
                            #x= torch.tensor([[1,0,0,0]]*3+[[0,1,0,0]]*3+[[0,0,1,0] if k in [0,1,2] else [0,0,0,1] for k in range(0,6) for i in range(1,tendrils_per_node+1) ]),
                            edge_index=torch.tensor(
                    [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]] + [[k, 6 * i + k] for i in
                                                                        range(1, tendrils_per_node + 1) for k in
                                                                        range(0, 6)]).transpose(0, 1),
                            y=torch.tensor([0]))
                data.y = torch.Tensor([0])


            data.num_nodes = n
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            idx += 1
            data_list.append(data)

        data,slices= self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])

    def len(self):
        return self.num_graphs
class SimpleRings(InMemoryDataset):
    def __init__(self, root, num_graphs= 100, transform=None, pre_transform=None):
        self.num_graphs= num_graphs
        self.name= "SimpleRings"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
       return ['ring_'+str(i)+'.pt' for i in range(self.num_graphs)]

    @property
    def processed_file_names(self):
        return ['ring_data.pt']

    #def download(self):
    #    # Download to `self.raw_dir`.
    #    path = download_url(url, self.raw_dir)
    #    ...

    def process(self):
        data_list= []
        idx = 0
        for n in range(int(self.num_graphs)):
            if n % 2==0:
                n=int(n/2) +6
                #for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                k=randrange(2,n-4+1)
                data = Data(x= torch.tensor([[1,0]]*n), edge_index=torch.tensor([[i,i+1] for i in range(k)]+[[k,0]]+[[i,i+1] for i in range(k+1,n-1)]+[[n-1,k+1]]).transpose(0,1),y= torch.tensor([0]))

                data.y= torch.Tensor([0])
                print("data0; ", data)
                print("data0.y", data.y)
                print("data0.x: ", data.x)
            else:
                n= int((n-1)/2)+6
                data= Data(x=torch.tensor([[1,0]]*n), edge_index=torch.tensor([[i,i+1] for i in range(n-1)]+[[n-1,0]]).transpose(0,1), y=torch.tensor([1]))
                data.y = torch.Tensor([1])

                print(data)
            data.num_nodes = n
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            idx += 1
            data_list.append(data)
        print(data_list)
        data,slices= self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])

    def len(self):
        return self.num_graphs
