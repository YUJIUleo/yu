"""
第一步：合成神经元图数据生成
运行方式：在 Anaconda Prompt 中执行 python generate_data.py
"""
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
import os

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 保存目录
os.makedirs("data", exist_ok=True)

def generate_neuron_graph(neuron_type, num_samples=20):
    """
    根据神经元类型生成合成图数据
    - HIP（锥体神经元）：分支多、深度大
    - DG（颗粒细胞）：分支少、结构紧凑
    - CB（浦肯野细胞）：扁平宽展开
    """
    graphs = []
    
    for i in range(num_samples):
        if neuron_type == "HIP":
            # 锥体神经元：深层二叉树，模拟丰富的树突分支
            depth = np.random.randint(4, 7)
            G = nx.balanced_tree(2, depth)
            # 添加一些随机长程连接模拟树突延伸
            nodes = list(G.nodes())
            for _ in range(int(len(nodes) * 0.1)):
                u, v = np.random.choice(nodes, 2, replace=False)
                G.add_edge(u, v)
            # 3D坐标：沿Y轴延伸（模拟锥体形态）
            pos = {}
            for node in G.nodes():
                pos[node] = [
                    np.random.randn() * 0.5,
                    node * 0.3 + np.random.randn() * 0.2,
                    np.random.randn() * 0.5
                ]
                
        elif neuron_type == "DG":
            # 颗粒细胞：浅层、紧凑
            depth = np.random.randint(2, 4)
            G = nx.balanced_tree(3, depth)
            # 3D坐标：紧凑球形分布
            pos = {}
            for node in G.nodes():
                pos[node] = np.random.randn(3) * 1.0
                
        elif neuron_type == "CB":
            # 浦肯野细胞：宽扁平面展开
            width = np.random.randint(5, 8)
            G = nx.balanced_tree(2, 3)
            # 添加横向连接模拟平面树突
            nodes = list(G.nodes())
            for _ in range(len(nodes)):
                u, v = np.random.choice(nodes, 2, replace=False)
                G.add_edge(u, v)
            # 3D坐标：XZ平面展开，Y方向压缩
            pos = {}
            for node in G.nodes():
                pos[node] = [
                    np.random.randn() * 3.0,
                    np.random.randn() * 0.3,
                    np.random.randn() * 3.0
                ]
        
        # 构建节点特征（3维坐标）
        num_nodes = G.number_of_nodes()
        x = torch.tensor([pos[n] for n in range(num_nodes)], dtype=torch.float32)
        
        # 构建边索引（无向图，双向边）
        edges = list(G.edges())
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # 标签编码：HIP=0, DG=1, CB=2
        label_map = {"HIP": 0, "DG": 1, "CB": 2}
        y = torch.tensor([label_map[neuron_type]], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
    
    return graphs

# 生成三类神经元数据
print("正在生成合成神经元数据...")
all_data = []

for neuron_type in ["HIP", "DG", "CB"]:
    graphs = generate_neuron_graph(neuron_type, num_samples=20)
    all_data.extend(graphs)
    print(f"  {neuron_type}: 生成 {len(graphs)} 个样本, "
          f"平均节点数={np.mean([g.num_nodes for g in graphs]):.0f}, "
          f"平均边数={np.mean([g.num_edges for g in graphs]):.0f}")

print(f"\n总计: {len(all_data)} 个图数据")

# 保存
torch.save(all_data, "data/all_graphs.pt")
print("已保存到 data/all_graphs.pt")
print("✅ 数据生成完成！")