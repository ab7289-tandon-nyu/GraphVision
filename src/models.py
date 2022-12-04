from typing import Any, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn

from utils import normalized_cut_2d


class SplineNet(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        conv_feat_list: List[int],
        kernel_size: int = 3,
        dropout: float = 0.0,
        fc_expansion: int = 2,
        use_cluster_pooling: bool = False,
        transform: Optional[Any] = None,
    ):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.use_cluster_pooling = use_cluster_pooling
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.fc_expansion = fc_expansion
        self.num_layers = len(conv_feat_list)
        self.transform = transform

        self.convs = nn.ModuleList()
        for idx in range(len(conv_feat_list)):
            if idx == 0:
                self.convs.append(
                    gnn.SplineConv(
                        self.input_features,
                        conv_feat_list[idx],
                        dim=2,
                        kernel_size=self.kernel_size,
                    )
                )
            else:
                self.convs.append(
                    gnn.SplineConv(
                        conv_feat_list[idx - 1],
                        conv_feat_list[idx],
                        dim=2,
                        kernel_size=self.kernel_size,
                    )
                )

        self.fc_list = nn.ModuleList()
        self.fc_list.append(
            gnn.Linear(conv_feat_list[-1], conv_feat_list[-1] * self.fc_expansion)
        )
        self.fc_list.append(
            gnn.Linear(conv_feat_list[-1] * self.fc_expansion, output_features)
        )

    def forward(self, data):
        x, batch = None

        for i in range(self.num_layers):
            is_last = i == self.num_layers - 1

            data.x = F.elu(self.convs[i](data.x, data.edge_index, data.edge_attr))

            if self.use_cluster_pooling and not is_last:
                data.edge_attr = None
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = gnn.graclus(data.edge_index, weight, data.x.size(0))
                data = gnn.max_pool(cluster, data, transform=self.transform)

            if is_last:
                data.edge_attr = None
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = gnn.graclus(data.edge_index, weight, data.x.size(0))
                x, batch = gnn.max_pool_x(cluster, data.x, data.batch)

        x = gnn.global_mean_pool(x, batch)
        x = F.elu(self.fc_list[0](x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc_list[1](x)
