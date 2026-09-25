"""Component 4 - Heterogeneous GNN model (XG-NID Sec. 3.1.4, Eqs. 5-9).

    h1 = ReLU(GATConv(h0, A, E))
    h2 = ReLU(BN(GATConv(h1, A, E)))
    h_graph = GlobalMeanPool(h2)                      # concatenated per node type
    out = LogSoftmax(W2 . ReLU(W1 . ReLU(W0 . h_graph)))

GATConv is used because attention lets the model decide which of a flow's 20
packets matter -- for a brute-force flow that is the one packet carrying the
credential exchange.

``forward`` takes plain dicts rather than a ``HeteroData`` object so that
Component 5 can differentiate the output with respect to ``x_dict`` directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, global_mean_pool

# Edge-attribute widths, from Eqs. 3 and 4.
DEFAULT_EDGE_DIMS = {
    ("flow", "contain", "packet"): 4,
    ("packet", "rev_contain", "flow"): 4,
    ("packet", "link", "packet"): 1,
}


class XGNIDHGNN(nn.Module):
    def __init__(
        self,
        metadata,
        num_classes: int,
        hidden: int = 64,
        heads: int = 1,
        edge_dims: dict | None = None,
        use_edge_attr: bool = True,
        dropout: float = 0.0,
        head_dims: tuple[int, ...] = (64, 16),
        bn_eps: float = 1e-5,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        edge_dims = edge_dims or DEFAULT_EDGE_DIMS

        def make_conv():
            return HeteroConv(
                {
                    et: GATConv(
                        (-1, -1), hidden, heads=heads, concat=False,
                        add_self_loops=False,
                        edge_dim=edge_dims.get(tuple(et)) if use_edge_attr else None,
                    )
                    for et in self.edge_types
                },
                aggr="sum",
            )

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        # Eq. 7 applies BN after the second GATConv; BN after the first as well
        # keeps the two branches (85-dim flow, 1500-dim payload) on one scale.
        self.bn1 = nn.ModuleDict({t: nn.BatchNorm1d(hidden, eps=bn_eps) for t in self.node_types})
        self.bn2 = nn.ModuleDict({t: nn.BatchNorm1d(hidden, eps=bn_eps) for t in self.node_types})
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

        dims = [hidden * len(self.node_types), *head_dims]
        layers: list[nn.Module] = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.head = nn.Sequential(*layers)

    def embed(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        """Graph-level embedding (Eq. 8), before the classification head."""
        kwargs = {"edge_attr_dict": edge_attr_dict} if self.use_edge_attr else {}
        x = self.conv1(x_dict, edge_index_dict, **kwargs)
        x = {k: self.act1(self.bn1[k](v)) for k, v in x.items()}
        x = self.conv2(x, edge_index_dict, **kwargs)
        x = {k: self.act2(self.bn2[k](v)) for k, v in x.items()}
        pooled = [global_mean_pool(x[t], batch_dict[t]) for t in self.node_types]
        return torch.cat(pooled, dim=1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        return F.log_softmax(self.head(self.embed(
            x_dict, edge_index_dict, edge_attr_dict, batch_dict)), dim=1)

    @staticmethod
    def loss(log_probs, target, weight=None):
        return F.nll_loss(log_probs, target, weight=weight)


def unpack(batch):
    """HeteroData batch -> the four dicts ``forward`` expects."""
    return (
        batch.x_dict,
        batch.edge_index_dict,
        {k: v for k, v in batch.edge_attr_dict.items()},
        batch.batch_dict,
    )


def build_model(sample, num_classes: int, **kwargs) -> XGNIDHGNN:
    """Instantiate and lazily initialise the model from one example graph."""
    from torch_geometric.loader import DataLoader

    model = XGNIDHGNN(sample.metadata(), num_classes, **kwargs)
    batch = next(iter(DataLoader([sample, sample], batch_size=2)))
    model.eval()
    with torch.no_grad():
        model(*unpack(batch))
    return model
