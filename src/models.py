from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.data import Data

from src.utils import normalized_cut_2d


def get_conv_layer(
    conv_type: str, hidden_features: Union[int,Tuple[int,int]], edge_dim: Optional[int] = None
):
    """
    Creates a GCN layer with the specified convolution type
    """

    in_features, out_features = None, None
    if isinstance(hidden_features, Tuple):
        in_features, out_features = hidden_features
    else:
        in_features, out_features = hidden_features, hidden_features

    if conv_type == "GEN":
        return gnn.GENConv(
            in_features,
            out_features,
            aggr="softmax",
            learn_t=True,
            edge_dim=edge_dim,
        )
    elif conv_type == "General":
        return gnn.GeneralConv(
            in_features, out_features, in_edge_channels=edge_dim
        )
    elif conv_type == "GAT":
        # TODO
        raise NotImplementedError()
    # add other cases

    else:
        raise ValueError(f"Invalid conv layer type: {conv_type}")


def get_act_layer(act_type: str):
    """
    Creates a activation layer given the specificed type
    """
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "elu":
        return nn.ELU()
    # add other cases

    else:
        raise ValueError(f"Invalid activation layer type: {act_type}")


def get_norm_layer(norm_type: str, hidden_features: int):
    """
    Returns a normalization layer based on the passed in type
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_features)
    elif norm_type == "layer":
        return nn.LayerNorm(hidden_features)
    # add other cases

    else:
        raise ValueError(f"Invalid normalization layer type: {norm_type}")


class DeeperGCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: Union[int, list[int]],
        conv_type: str = "GEN",
        act: str = "relu",
        norm: str = "batch",
        num_layers: int = 2,
        use_cluster_pooling: bool = False,
        readout: str = "mean",
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.readout = readout
        self.use_cluster_pooling = use_cluster_pooling
        self.dropout = dropout
        
        if isinstance(hidden_features, list):
            if len(hidden_features) != num_layers:
                raise ValueError(f"hidden layers length of {len(hidden_features)} must match number of layers {num_layers}")
        elif isinstance(hidden_features, int):
            hidden_features = [hidden_features] * num_layers    

        self.fc_in = nn.Linear(input_features, hidden_features[0])
        self.fc_out = nn.Linear(hidden_features[-1], output_features)
        self.out_act = get_act_layer(act)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                gnn.DeepGCNLayer(
                    conv=get_conv_layer(conv_type, (hidden_features[i], hidden_features[i]), edge_dim=edge_dim),
                    act=get_act_layer(act),
                    norm=get_norm_layer(norm, hidden_features[i]),
                    block="res+",
                    dropout=dropout,
                )
            )

    def forward(self, data: Data) -> torch.Tensor:
        data.x = self.fc_in(data.x)

        for layer in self.layers:
            data.x = layer(data.x, data.edge_index, data.edge_attr)

        x, batch = data.x, data.batch
        if self.use_cluster_pooling:
            data.edge_attr = None
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = gnn.graclus(data.edge_index, weight, data.x.size(0))
            x, batch = gnn.max_pool_x(cluster, x, batch)

        if self.readout == "mean":
            x = gnn.global_mean_pool(x, batch)
        elif self.readout == "max":
            x = gnn.global_max_pool(x, batch)
        elif self.readout == "add":
            x = gnn.global_add_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_act(x)

        return self.fc_out(x)
