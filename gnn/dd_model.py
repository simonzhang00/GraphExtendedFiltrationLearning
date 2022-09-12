import torch
import numpy as np
import torch.nn as nn

import functools

from torchPHext.torchex_PHext.nn import SLayerRationalHat, SLayerSquare, SLayerExponential
from torch_geometric.nn import SAGEConv, LEConv, GINConv, global_add_pool, global_sort_pool, global_mean_pool, \
    global_max_pool, Set2Set

from torchPHext.torchex_PHext import pershom as pershom_ext
from torch_geometric.nn import GINConv, global_add_pool
import sys

sys.path.append(".")

ph_extended_link_tree = pershom_ext.pershom_backend.__C.VertExtendedFiltCompCuda_link_cut_tree__extended_persistence_batch
ph_extended_link_tree_cyclereps= pershom_ext.pershom_backend.__C.VertExtendedFiltCompCuda_link_cut_tree_cyclereps__extended_persistence_batch

def gin_mlp_factory(gin_mlp_type: str, dim_in: int, dim_out: int):
    if gin_mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif gin_mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )

    elif gin_mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )
    else:
        raise ValueError("Unknown gin_mlp_type!")


def ClassifierHead(
        dataset,
        dim_in: int = None,
        hidden_dim: int = None,
        drop_out: float = None):
    assert (0.0 <= drop_out) and (drop_out < 1.0)
    assert dim_in is not None
    assert drop_out is not None
    assert hidden_dim is not None

    tmp = [
        nn.Linear(dim_in, hidden_dim),
        nn.LeakyReLU(),
    ]

    if drop_out > 0:
        tmp += [nn.Dropout(p=drop_out)]

    tmp += [nn.Linear(hidden_dim, dataset.num_classes)]

    return nn.Sequential(*tmp)

class DegreeOnlyFiltration(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        tmp = []
        for i, j in zip(batch.sample_pos[:-1], batch.sample_pos[1:]):
            max_deg = batch.node_deg[i:j].max()

            t = torch.ones(j - i, dtype=torch.float, device=batch.node_deg.device)
            t = t * max_deg
            tmp.append(t)

        max_deg = torch.cat(tmp, dim=0)

        normalized_node_deg = batch.node_deg.float() / max_deg

        return normalized_node_deg

class ClassicGNN(torch.nn.Module):
    def __init__(self,
                 dataset,
                 use_node_degree=None,
                 set_node_degree_uninformative=None,
                 use_node_label=None,
                 use_raw_node_label=None,
                 gin_number=None,
                 gin_dimension=None,
                 conv_type='GIN',
                 gin_mlp_type=None,
                 cls_hidden_dimension=512,
                 drop_out=0.5,
                 **kwargs
                 ):
        super().__init__()

        dim = gin_dimension

        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab

        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = nn.Embedding(max_node_deg + 1, dim)
        else:
            self.embed_deg = None

        self.embed_lab = nn.Embedding(num_node_lab, dim) if use_node_label and False else None
        self.use_raw_node_label = use_raw_node_label

        dim_input = dim*((self.embed_deg is not None) + (self.embed_lab is not None))
        if (use_raw_node_label):
            dim_input += dataset[0].x.size(1)

        dims = [dim_input] + (gin_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu
        if use_node_label and False:
            self.node_label_embedder = torch.nn.Linear(num_node_lab, gin_dimension)

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            if conv_type == 'GIN':
                l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
                self.convs.append(GINConv(l, train_eps=True))
            elif conv_type == 'GraphSAGE':
                self.convs.append(SAGEConv(n_1, n_2))

            self.bns.append(nn.BatchNorm1d(n_2))
        self.fc = None
        self.classifier_extpers = ClassifierHead(dataset, gin_dimension, int(cls_hidden_dimension),
                                                 drop_out=drop_out)

    def forward2(self, h, batch):
        h = global_add_pool(h, batch.batch)
        out = self.classifier_extpers(h)
        out = torch.nn.LogSoftmax(dim=1)(out)
        return out

    def forward(self, batch, device):

        node_deg = batch.node_deg
        if hasattr(batch, "node_lab"):
            node_lab = batch.node_lab
        else:
            node_lab = None
        edge_index = batch.edge_index
        if self.use_raw_node_label:
            if self.embed_deg is not None:
                tmp = [self.embed_deg(node_deg), batch.x]
            else:
                tmp = []
                raise ValueError
        else:
            tmp = [e(x) for e, x in
                   zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
                   if e is not None]

        if len(tmp) > 0:
            tmp = torch.cat(tmp, dim=1)
        else:
            tmp = torch.tensor(tmp)
        x = tmp
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.act(x)
            x = torch.nn.Dropout(p=0.5)(x)
        x = self.convs[-1](x, edge_index)

        return x


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


class Filtration(torch.nn.Module):
    def __init__(self,
                 dataset,
                 use_node_degree=None,
                 set_node_degree_uninformative=None,
                 use_node_label=None,
                 use_raw_node_label=None,
                 filt_conv_number=None,
                 filt_conv_dimension=None,
                 gin_mlp_type=None,
                 **kwargs
                 ):
        super().__init__()

        dim = filt_conv_dimension

        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab

        if use_node_degree:
            self.embed_deg = nn.Embedding(max_node_deg + 1, dim)
        else:
            self.embed_deg = None

        self.embed_lab = nn.Embedding(num_node_lab, dim) if (use_node_label and not use_raw_node_label and not (num_node_lab==0 or num_node_lab is None)) else None
        self.use_raw_node_label = use_raw_node_label

        dim_input = dim*((self.embed_deg is not None) + (self.embed_lab is not None))

        if (use_raw_node_label) and dataset[0].x is not None:
            dim_input += dataset[0].x.size(1)

        dims = [dim_input] + (filt_conv_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu
        if use_raw_node_label and dataset[0].x is not None:
            self.node_label_embedder = torch.nn.Linear(dataset[0].x.size(1), filt_conv_dimension)

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        self.fc = nn.Sequential(
            nn.Linear(sum(dims), dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        if dim_input==0:
            raise Exception("Cannot have neither degree nor node features")
        elif 0 in dims:
            raise Exception("Cannot have 0 dim hidden layers")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc.apply(weight_reset)

    def forward(self, batch):
        node_deg = batch.node_deg
        if hasattr(batch, "node_lab"):
            node_lab = batch.node_lab
        else:
            node_lab = None
        edge_index = batch.edge_index
        if self.use_raw_node_label:
            if self.embed_deg is not None and self.fc is not None:
                tmp = [self.embed_deg(node_deg), batch.x]
            else:
                tmp = [batch.x]
        else:
            tmp = [e(x) for e, x in
                   zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
                   if e is not None]

        tmp = torch.cat(tmp, dim=1)

        z = [tmp]

        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)

        x = torch.cat(z, dim=1)
        ret = self.fc(x).squeeze()
        return ret


class StandardPershomReadout(nn.Module):
    def __init__(self,
                 dataset,
                 num_struct_elements=None,
                 cls_hidden_dimension=None,
                 drop_out=None,
                 ):
        super().__init__()
        assert isinstance(num_struct_elements, int)

        self.ldgm_0 = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        self.ldgm_0_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        self.ldgm_1_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        fc_in_feat = num_struct_elements

        self.cls_head = ClassifierHead(
            dataset,
            dim_in=fc_in_feat,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    def forward(self, h_0, h_0_ess, h_1_ess):
        tmp = []

        tmp.append(self.ldgm_0(h_0))
        tmp.append(self.ldgm_0_ess(h_0_ess))
        tmp.append(self.ldgm_1_ess(h_1_ess))
        x = torch.cat(tmp, dim=1)
        return x


class PershomReadout(nn.Module):
    def __init__(self,
                 dataset,
                 num_struct_elements=None,
                 cls_hidden_dimension=None,
                 drop_out=None,
                 ):
        super().__init__()
        assert isinstance(num_struct_elements, int)

        self.ldgm_0_up = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        self.ldgm_0_down = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        self.ldgm_cc = SLayerRationalHat(num_struct_elements, 2,
                                         radius_init=0.1)
        self.ldgm_h1 = SLayerRationalHat(num_struct_elements, 2,
                                         radius_init=0.1)

        self.ldgm_0_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        self.ldgm_1_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        fc_in_feat = num_struct_elements

        self.cls_head = ClassifierHead(
            dataset,
            dim_in=fc_in_feat,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    def forward(self, h_0_up, h_0_down, h_0_cc, h_1):
        tmp = []

        tmp.append(self.ldgm_0_up(h_0_up))
        tmp.append(self.ldgm_0_down(h_0_down))
        tmp.append(self.ldgm_cc(h_0_cc))
        tmp.append(self.ldgm_h1(h_1))

        x = torch.cat(tmp, dim=1)
        return x


class PershomClassifier(nn.Module):
    def __init__(self,
                 dataset,
                 fc_in_feat=None,
                 cls_hidden_dimension=None,
                 drop_out=None,
                 ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.cls_head = ClassifierHead(
            dataset,
            dim_in=fc_in_feat,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    def forward(self, x):
        if not self.use_as_feature_extractor:
            x = self.cls_head(x)

        return x


class PershomBase(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.readout_type = "extph"
        self.supervised = True
        self.use_super_level_set_filtration = None
        self.use_as_feature_extractor = False
        self.cls = None
        self.augmentor = aug

        self.convs = torch.nn.ModuleList()
        self.mlp = None
        self.classifier_gnn = None
        self.classifier_extpers = None
        self.classifier_standardpers = None
        self.repr2structelems = None
        self.structelems2repr = None
        self.standard_ph_readout = None
        self.epochs = 1
        self.p = 0.01
        self.gnn_node = None
        self.set2set = Set2Set(1, processing_steps=4)
        self.k= 5

    def compute_extended_ph_link_tree(self, node_filt, batch, device):
        ph_input = []
        for idx, (i, j, e) in enumerate(zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_info)):
            v = node_filt[i:j]  # extract vertex values
            v.to("cpu")
            e.to("cpu")
            # use this for visualization
            if idx == 0 and len(batch.boundary_info) == 1:
                # print("v: ",v)
                # print("edge_index: ", batch.edge_index)
                # print("x: ", batch.x)
                pass
            ph_input.append((v, [e]))
        pers = ph_extended_link_tree(ph_input)  # ph_input needs to be: (v,[e])

        h_0_up = [torch.stack([x.to(device) for x in g[0]]) for g in pers]
        h_0_down = [torch.stack([x.to(device) for x in g[1]]) for g in pers]
        h_0_extplus = [torch.stack([x.to(device) for x in g[2]]) for g in pers]
        h_1_extminus = [torch.stack([x.to(device) for x in g[3]]) for g in pers]

        return h_0_up, h_0_down, h_0_extplus, h_1_extminus

    def compute_extended_ph_link_tree_wcyclereps(self,node_filt, batch, device):
        ph_input= []
        for idx, (i, j, e) in enumerate(zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_info)):
            v = node_filt[i:j]#extract vertex values
            v.to("cpu")
            e.to("cpu")
            #use this for visualization
            if idx==0 and len(batch.boundary_info)==1:
                #print("v: ",v)
                #print("edge_index: ", batch.edge_index)
                #print("x: ", batch.x)
                pass
            ph_input.append((v, [e]))
        # ph_input needs to be: (v,[e])
        out= ph_extended_link_tree_cyclereps(ph_input)
        pers= [per[0] for per in out]
        cycle_reps = [cycles[1] for cycles in out]

        h_0_up= [torch.stack([x.to(device) for x in g[0]]) for g in pers]
        h_0_down= [torch.stack([x.to(device) for x in g[1]]) for g in pers]
        h_0_extplus= [torch.stack([x.to(device) for x in g[2]]) for g in pers]
        h_1_extminus= [torch.stack([x.to(device) for x in g[3]]) for g in pers]
        cycle_reps= [[torch.stack([x.to(device).unsqueeze(0) for x in c]) for c in cycle] for cycle in cycle_reps]

        return h_0_up, h_0_down, h_0_extplus, h_1_extminus, cycle_reps

    def forward(self, batch, device):
        assert self.use_super_level_set_filtration is not None
        if batch.x is not None:
            idx = torch.empty((batch.x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.p
            batch.x[:, idx] = 0
        node_filt0 = self.fil(batch)
        if self.readout_type == "extph":
            h_0_up_1, h_0_down_1, h_0_cc_1, h_1_1 = self.compute_extended_ph_link_tree(node_filt0, batch, device)
            g1 = self.readout(h_0_up_1, h_0_down_1, h_0_cc_1, h_1_1)
        elif self.readout_type == "extph_cyclereps":
            h_0_up_1, h_0_down_1, h_0_cc_1, h_1_1, cycle_reps = self.compute_extended_ph_link_tree_wcyclereps(node_filt0, batch, device)
            if self.use_bars:
                g1 = self.readout(h_0_up_1, h_0_down_1, h_0_cc_1, h_1_1)

            cycle_batch_reps = []
            for g in range(len(cycle_reps)):
                cyclegraph_reps = []
                for c in range(len(cycle_reps[g])):
                    output, (_, _) = self.lstm(cycle_reps[g][c].unsqueeze(0))
                    cyclegraph_reps.append(
                        torch.sum(torch.stack([output[0][i] for i in range(output.size(1))], dim=0), dim=0).unsqueeze(
                            0))
                graph_rep_var, graph_rep_mean = torch.var_mean(torch.cat(cyclegraph_reps, dim=0), dim=0)

                cyclegraph_reps = graph_rep_mean
                cycle_batch_reps.append(cyclegraph_reps.unsqueeze(0))
            cycle_batch_reps = torch.cat(cycle_batch_reps, dim=0)

            if self.use_bars:
                g1 = g1 + cycle_batch_reps
            else:
                g1= cycle_batch_reps
        elif self.readout_type == "sum":
            node_filt0 = node_filt0.unsqueeze(1)
            g1 = global_add_pool(node_filt0, batch.batch)
        elif self.readout_type == "max":
            node_filt0 = node_filt0.unsqueeze(1)
            g1 = global_max_pool(node_filt0, batch.batch)
        elif self.readout_type == "average":
            node_filt0 = node_filt0.unsqueeze(1)
            g1 = global_mean_pool(node_filt0, batch.batch)
        elif self.readout_type == "sort":
            node_filt0 = node_filt0.unsqueeze(1)
            g1 = global_sort_pool(node_filt0, batch.batch, self.k)
        elif self.readout_type == "set2set":
            node_filt0 = node_filt0.unsqueeze(1)
            g1 = self.set2set(node_filt0,batch.batch)
        out = g1
        out = self.classifier_gnn(out)
        out = torch.nn.LogSoftmax(dim=1)(out)
        return out

    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features

    @property
    def use_as_feature_extractor(self):
        return self.use_as_feature_extractor

    @use_as_feature_extractor.setter
    def use_as_feature_extractor(self, val):
        if hasattr(self, 'cls'):
            self.cls.use_as_feature_extractor = val

    def init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.bias.data.fill_(0.01)

        self.apply(init)


class ClassicReadoutFilt(PershomBase):
    def __init__(self,
                 dataset,
                 use_super_level_set_filtration: bool = None,
                 use_node_degree: bool = None,
                 set_node_degree_uninformative: bool = None,
                 use_node_label: bool = None,
                 use_raw_node_label: bool = None,
                 filt_conv_number: int = None,
                 filt_conv_dimension: int = None,
                 gin_mlp_type: str = None,
                 num_struct_elements: int = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 conv_number: int = None,
                 conv_dimension: int = None,
                 augmentor=(None, None),
                 readout="sum",
                 **kwargs,
                 ):
        super().__init__(augmentor)
        self.readout_type = readout

        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = Filtration(
            dataset,
            use_node_degree=use_node_degree,
            set_node_degree_uninformative=set_node_degree_uninformative,
            use_node_label=use_node_label,
            use_raw_node_label=use_raw_node_label,
            filt_conv_number=filt_conv_number,
            filt_conv_dimension=filt_conv_dimension,
            gin_mlp_type=gin_mlp_type,
        )

        self.standard_ph_readout = StandardPershomReadout(
            dataset,
            num_struct_elements=num_struct_elements,
            cls_hidden_dimension=cls_hidden_dimension,
            drop_out=drop_out
        )

        self.cls = PershomClassifier(dataset,
                                     fc_in_feat=num_struct_elements,
                                     cls_hidden_dimension=cls_hidden_dimension,
                                     drop_out=drop_out
                                     )
        self.gnn = ClassicGNN(dataset,
                              use_node_degree=use_node_degree,
                              set_node_degree_uninformative=set_node_degree_uninformative,
                              use_node_label=use_node_label,
                              use_raw_node_label=use_raw_node_label,
                              gin_number=conv_number,
                              conv_type='GIN',
                              gin_dimension=conv_dimension,
                              gin_mlp_type=gin_mlp_type,
                              )
        self.supervised = True
        self.k= int(np.percentile([d.num_nodes for d in dataset], 10))
        if self.readout_type == "sort":
            self.classifier_gnn = ClassifierHead(dataset, self.k, cls_hidden_dimension, drop_out=drop_out)
        elif self.readout_type== "set2set":
            self.classifier_gnn = ClassifierHead(dataset, 2, cls_hidden_dimension, drop_out=drop_out)
        else:
            self.classifier_gnn = ClassifierHead(dataset, 1, cls_hidden_dimension, drop_out=drop_out)
        self.init_weights()

    def reset_parameters(self):
        self.fil.reset_parameters()
        self.classifier_gnn.apply(weight_reset)


class PershomLearnedFilt(PershomBase):
    def __init__(self,
                 dataset,
                 use_super_level_set_filtration: bool = None,
                 use_node_degree: bool = None,
                 set_node_degree_uninformative: bool = None,
                 use_node_label: bool = None,
                 use_raw_node_label: bool = None,
                 gin_number: int = None,
                 gin_dimension: int = None,
                 gin_mlp_type: str = None,
                 num_struct_elements: int = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 augmentor=(None, None),
                 readout="extph",
                 **kwargs,
                 ):
        super().__init__(augmentor)

        self.readout_type = readout
        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = Filtration(
            dataset,
            use_node_degree=use_node_degree,
            set_node_degree_uninformative=set_node_degree_uninformative,
            use_node_label=use_node_label,
            use_raw_node_label=use_raw_node_label,
            gin_number=gin_number,
            gin_dimension=gin_dimension,
            gin_mlp_type=gin_mlp_type,
        )

        self.readout = PershomReadout(
            dataset,
            num_struct_elements=num_struct_elements,
            cls_hidden_dimension=cls_hidden_dimension,
            drop_out=drop_out
        )
        self.readout.use_as_feature_extractor = True
        self.cls = PershomClassifier(dataset,
                                     fc_in_feat=num_struct_elements,
                                     cls_hidden_dimension=cls_hidden_dimension,
                                     drop_out=drop_out
                                     )
        self.gnn = ClassicGNN(dataset,
                              use_node_degree=use_node_degree,
                              set_node_degree_uninformative=set_node_degree_uninformative,
                              use_node_label=use_node_label,
                              use_raw_node_label=use_raw_node_label,
                              gin_number=gin_number,
                              gin_dimension=gin_dimension,
                              gin_mlp_type=gin_mlp_type,
                              )

        self.init_weights()
        self.supervised = False


class PershomLearnedFiltSup(PershomBase):
    def __init__(self,
                 dataset,
                 use_super_level_set_filtration: bool = None,
                 use_node_degree: bool = None,
                 set_node_degree_uninformative: bool = None,
                 use_node_label: bool = None,
                 use_raw_node_label: bool = None,
                 filt_conv_number: int = None,
                 filt_conv_dimension: int = None,
                 gin_mlp_type: str = None,
                 num_struct_elements: int = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 conv_number: int = None,
                 conv_dimension: int = None,
                 augmentor=(None, None),
                 readout="extph",
                 use_bars= True,
                 **kwargs,
                 ):
        super().__init__(augmentor)
        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = Filtration(
            dataset,
            use_node_degree=use_node_degree,
            set_node_degree_uninformative=set_node_degree_uninformative,
            use_node_label=use_node_label,
            use_raw_node_label=use_raw_node_label,
            filt_conv_number=filt_conv_number,
            filt_conv_dimension=filt_conv_dimension,
            gin_mlp_type=gin_mlp_type,
        )

        self.standard_ph_readout = StandardPershomReadout(
            dataset,
            num_struct_elements=num_struct_elements,
            cls_hidden_dimension=cls_hidden_dimension,
            drop_out=drop_out
        )

        self.readout = PershomReadout(
            dataset,
            num_struct_elements=num_struct_elements,
            cls_hidden_dimension=cls_hidden_dimension,
            drop_out=drop_out
        )
        self.readout.use_as_feature_extractor = True
        self.cls = PershomClassifier(dataset,
                                     fc_in_feat=num_struct_elements,
                                     cls_hidden_dimension=cls_hidden_dimension,
                                     drop_out=drop_out
                                     )
        self.gnn = ClassicGNN(dataset,
                              use_node_degree=use_node_degree,
                              set_node_degree_uninformative=set_node_degree_uninformative,
                              use_node_label=use_node_label,
                              use_raw_node_label=use_raw_node_label,
                              gin_number=conv_number,
                              conv_type='GIN',
                              gin_dimension=conv_dimension,
                              gin_mlp_type=gin_mlp_type,
                              )
        self.classifier_gnn = ClassifierHead(dataset, 4 * num_struct_elements, cls_hidden_dimension, drop_out=drop_out)
        self.classifier_extpers = ClassifierHead(dataset, 4 * num_struct_elements, int(cls_hidden_dimension),
                                                 drop_out=drop_out)
        self.classifier_standardpers = ClassifierHead(dataset, 3 * num_struct_elements, int(cls_hidden_dimension),
                                                      drop_out=drop_out)

        self.supervised = True
        self.repr2structelems = nn.Linear(conv_dimension, 4 * num_struct_elements)
        self.structelems2repr = nn.Linear(4 * num_struct_elements, conv_dimension)
        self.use_bars= use_bars
        self.init_weights()

    def reset_parameters(self):
        self.fil.reset_parameters()
        self.classifier_gnn.apply(weight_reset)


class PershomRigidDegreeFilt(PershomBase):
    def __init__(self,
                 dataset,
                 use_super_level_set_filtration: bool = None,
                 num_struct_elements: int = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 **kwargs,
                 ):
        super().__init__()

        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = DegreeOnlyFiltration()

        self.cls = PershomClassifier(
            dataset,
            drop_out=drop_out,
            cls_hidden_dimension=cls_hidden_dimension
        )

        self.init_weights()
        self.supervised = False


class OneHotEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        eye = torch.eye(dim, dtype=torch.float)

        self.register_buffer('eye', eye)

    def forward(self, batch):
        assert batch.dtype == torch.long

        return self.eye.index_select(0, batch)

    @property
    def dim(self):
        return self.eye.size(1)


class UniformativeDummyEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        b = torch.ones(1, dim, dtype=torch.float)
        self.register_buffer('ones', b)

    def forward(self, batch):
        assert batch.dtype == torch.long
        return self.ones.expand(batch.size(0), -1)

    @property
    def dim(self):
        return self.ones.size(1)


class GIN(nn.Module):
    def __init__(self,
                 dataset,
                 use_node_degree: bool = None,
                 use_node_label: bool = None,
                 gin_number: int = None,
                 gin_dimension: int = None,
                 gin_mlp_type: str = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 set_node_degree_uninformative: bool = None,
                 pooling_strategy: str = None,
                 **kwargs,
                 ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension

        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab

        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg + 1)
        else:
            self.embed_deg = None

        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None

        dim_input = 0
        dim_input += self.embed_deg.dim if use_node_degree else 0
        dim_input += self.embed_lab.dim if use_node_label else 0
        assert dim_input > 0

        dims = [dim_input] + (gin_number) * [dim]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
        else:
            raise ValueError

        self.cls = ClassifierHead(
            dataset,
            dim_in=gin_dimension,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features

    def forward(self, batch):

        node_deg = batch.node_deg
        node_lab = batch.node_lab

        edge_index = batch.edge_index

        tmp = [e(x) for e, x in
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None]

        tmp = torch.cat(tmp, dim=1)

        z = [tmp]

        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)

        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            x = self.sort_pool_nn(x)
            x = x.squeeze()

        if not self.use_as_feature_extractor:
            x = self.cls(x)

        return x


class SimpleNNBaseline(nn.Module):
    def __init__(self,
                 dataset,
                 use_node_degree: bool = None,
                 use_node_label: bool = None,
                 set_node_degree_uninformative: bool = None,
                 gin_dimension: int = None,
                 gin_mlp_type: str = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 pooling_strategy: str = None,
                 **kwargs,
                 ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension

        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab

        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg + 1)
        else:
            self.embed_deg = None

        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None

        dim_input = 0
        dim_input += self.embed_deg.dim if use_node_degree else 0
        dim_input += self.embed_lab.dim if use_node_label else 0
        assert dim_input > 0

        self.mlp = gin_mlp_factory(gin_mlp_type, dim_input, dim)

        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
        else:
            raise ValueError

        self.cls = ClassifierHead(
            dataset,
            dim_in=gin_dimension,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features

    def forward(self, batch):

        node_deg = batch.node_deg
        node_lab = batch.node_lab

        edge_index = batch.edge_index

        tmp = [e(x) for e, x in
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None]

        x = torch.cat(tmp, dim=1)

        x = self.mlp(x)
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            x = self.sort_pool_nn(x)
            x = x.squeeze()

        if not self.use_as_feature_extractor:
            x = self.cls(x)

        return x

