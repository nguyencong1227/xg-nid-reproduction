#!/usr/bin/env python
"""In shape cua tat ca bien/tham so trong Eq.5, 6, 7 cua XG-NID (Sec 3.1.4) tren 1 sample that.

    Eq.5:  h_v^(1) = ReLU( GATConv(h_v^(0), A, E) )
    Eq.6:  h_i^(l) = sigma( sum_j alpha_ij W h_j + sum_e alpha_ij W_e h_e )
    Eq.7:  h_v^(2) = ReLU( BN( GATConv(h_v^(1), A, E) ) )

Usage:
    python scripts/inspect_eq567.py                       # sample dau tien co 20 goi
    python scripts/inspect_eq567.py --index 4              # chon dung graph index
    python scripts/inspect_eq567.py --run work/xgnid/05_run --graphs work/xgnid/04_graphs
"""
import argparse
import sys
sys.path.insert(0, ".")

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from xgnid.step3_graph import XGNIDGraphDataset
from xgnid.step4_model import XGNIDHGNN, unpack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work/xgnid/05_run")
    ap.add_argument("--graphs", default="work/xgnid/04_graphs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=None, help="graph index; mac dinh: sample dau tien co 20 goi")
    args = ap.parse_args()

    ds = XGNIDGraphDataset(root=f"{args.graphs}/{args.split}", name=args.split)
    idx = args.index if args.index is not None else next(
        i for i in range(len(ds)) if ds[i]["packet"].x.shape[0] == 20)
    sample = ds[idx]
    print(f"### Sample graph_index={idx}: {sample}\n")

    ck = torch.load(f"{args.run}/model.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = XGNIDHGNN(ck["metadata"], len(ck["class_names"]), hidden=cfg["hidden"], heads=cfg["heads"],
                      use_edge_attr=cfg["use_edge_attr"], dropout=cfg["dropout"], head_dims=tuple(cfg["head_dims"]))
    warm = next(iter(DataLoader([sample, sample], batch_size=2)))
    model.eval()
    with torch.no_grad():
        model(*unpack(warm))          # materialise lazy GATConv shapes
    model.load_state_dict(ck["state_dict"])
    model.eval()

    batch = next(iter(DataLoader([sample], batch_size=1)))
    x_dict, edge_index_dict, edge_attr_dict, batch_dict = unpack(batch)

    print("=" * 70)
    print("Eq.5:  h_v^(1) = ReLU( GATConv( h_v^(0), A, E ) )")
    print("=" * 70)
    print("--- h_v^(0)  (node feature dau vao, x_dict) ---")
    for k, v in x_dict.items():
        print(f"    h_{k}^(0)  : {tuple(v.shape)}")

    print("\n--- A  (edge_index_dict, moi loai canh 1 shape) ---")
    for k, v in edge_index_dict.items():
        print(f"    A[{k}]  : {tuple(v.shape)}")

    print("\n--- E  (edge_attr_dict, moi loai canh 1 shape) ---")
    for k, v in edge_attr_dict.items():
        print(f"    E[{k}]  : {tuple(v.shape)}")

    with torch.no_grad():
        h1_pre = model.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
    print("\n--- sau GATConv (truoc ReLU) ---")
    for k, v in h1_pre.items():
        print(f"    GATConv(h_{k}^(0))  : {tuple(v.shape)}")

    with torch.no_grad():
        h1 = {k: model.act1(model.bn1[k](v)) for k, v in h1_pre.items()}
    print("\n--- h_v^(1)  (sau BN+LeakyReLU cua conv1) ---")
    for k, v in h1.items():
        print(f"    h_{k}^(1)  : {tuple(v.shape)}")

    print("\n" + "=" * 70)
    print("Eq.6: chi tiet 1 GATConv rieng le - lay canh (flow,contain,packet)")
    print("      alpha_ij = attention coeff; W = lin_src/lin_dst; W_e = lin_edge")
    print("=" * 70)
    contain_conv: GATConv = model.conv1.convs[("flow", "contain", "packet")]
    print(f"    W  (lin_src.weight)  : {tuple(contain_conv.lin_src.weight.shape)}   <- W^(l), nhan h_j (flow node)")
    print(f"    W  (lin_dst.weight)  : {tuple(contain_conv.lin_dst.weight.shape)}   <- nhan h_i (packet node, dst)")
    print(f"    W_e (lin_edge.weight): {tuple(contain_conv.lin_edge.weight.shape)}   <- W_e^(l), nhan h_e (edge attr)")
    print(f"    att_src               : {tuple(contain_conv.att_src.shape)}")
    print(f"    att_dst               : {tuple(contain_conv.att_dst.shape)}")
    print(f"    att_edge               : {tuple(contain_conv.att_edge.shape)}")

    with torch.no_grad():
        out_c, (edge_idx_c, alpha) = contain_conv(
            (x_dict["flow"], x_dict["packet"]),
            edge_index_dict[("flow", "contain", "packet")],
            edge_attr=edge_attr_dict[("flow", "contain", "packet")],
            return_attention_weights=True,
        )
    print(f"\n    alpha_ij (attention weight moi canh contain): {tuple(alpha.shape)}")
    print(f"    5 gia tri alpha_ij dau tien: {alpha[:5].flatten().tolist()}")
    print(f"    tong alpha_ij (moi packet chi co 1 canh contain -> luon =1.0): {alpha.sum().item():.4f}")
    print(f"    output GATConv cho canh nay: {tuple(out_c.shape)}   <- [n_packet, hidden]")

    print("\n" + "=" * 70)
    print("Eq.7:  h_v^(2) = ReLU( BN( GATConv( h_v^(1), A, E ) ) )")
    print("=" * 70)
    with torch.no_grad():
        h2_pre = model.conv2(h1, edge_index_dict, edge_attr_dict=edge_attr_dict)
    print("--- sau GATConv lan 2 (truoc BN) ---")
    for k, v in h2_pre.items():
        print(f"    GATConv(h_{k}^(1))  : {tuple(v.shape)}")

    with torch.no_grad():
        h2_bn = {k: model.bn2[k](v) for k, v in h2_pre.items()}
    print("\n--- sau BatchNorm (truoc LeakyReLU) ---")
    for k, v in h2_bn.items():
        print(f"    BN(h_{k}^(2))  : {tuple(v.shape)}")

    with torch.no_grad():
        h2 = {k: model.act2(v) for k, v in h2_bn.items()}
    print("\n--- h_v^(2)  (dau ra cuoi Eq.7) ---")
    for k, v in h2.items():
        print(f"    h_{k}^(2)  : {tuple(v.shape)}")


if __name__ == "__main__":
    main()
