"""
第三步：Streamlit 展示平台
运行方式：streamlit run app.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import os

# ==================== 页面配置 ====================
st.set_page_config(page_title="神经元GCN对比学习平台", page_icon="🧠", layout="wide")

# ==================== 模型定义（与训练时一致） ====================
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        h = global_mean_pool(h, batch)
        return F.normalize(h, p=2, dim=-1)

# ==================== 加载数据 ====================
@st.cache_resource
def load_data():
    all_data = torch.load("data/all_graphs.pt", weights_only=False)
    return all_data

@st.cache_resource
def load_model():
    model = GCNEncoder(in_dim=3, hidden_dim=32, out_dim=16)
    if os.path.exists("models/gcn_encoder.pth"):
        model.load_state_dict(torch.load("models/gcn_encoder.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model

all_data = load_data()
model = load_model()

# 提取所有嵌入
@st.cache_data
def get_all_embeddings():
    with torch.no_grad():
        loader = Batch.from_data_list(all_data)
        emb = model(loader.x, loader.edge_index, loader.batch)
    return emb.numpy()

embeddings = get_all_embeddings()
labels = np.array([d.y.item() for d in all_data])
type_names = {0: "HIP(锥体神经元)", 1: "DG(颗粒细胞)", 2: "CB(浦肯野细胞)"}
colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

# ==================== 侧边栏导航 ====================
st.sidebar.title("🧠 神经元对比学习平台")
page = st.sidebar.radio("选择页面", ["📊 数据概览", "📈 训练结果", "🔬 嵌入可视化", "🔮 预测演示"])

# ==================== 页面1：数据概览 ====================
if page == "📊 数据概览":
    st.title("📊 数据概览")
    
    col1, col2, col3 = st.columns(3)
    for label_id, name in type_names.items():
        count = np.sum(labels == label_id)
        avg_nodes = np.mean([all_data[i].num_nodes for i in range(len(all_data)) if labels[i] == label_id])
        avg_edges = np.mean([all_data[i].num_edges for i in range(len(all_data)) if labels[i] == label_id])
        with st.container():
            st.subheader(f"{name}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("样本数", count)
            mc2.metric("平均节点数", f"{avg_nodes:.0f}")
            mc3.metric("平均边数", f"{avg_edges:.0f}")
    
    st.subheader("数据集统计")
    st.write(f"- 总样本数: **{len(all_data)}**")
    st.write(f"- 节点特征维度: **3** (x, y, z 坐标)")
    st.write(f"- 神经元类型: **3** 类 (HIP / DG / CB)")
    st.write(f"- 嵌入维度: **16**")
    
    # 画一个示例神经元的3D图
    st.subheader("示例神经元3D结构")
    sample_type = st.selectbox("选择神经元类型", ["HIP", "DG", "CB"])
    type_id = {"HIP": 0, "DG": 1, "CB": 2}[sample_type]
    sample_idx = np.where(labels == type_id)[0][0]
    sample = all_data[sample_idx]
    
    fig = go.Figure()
    edge_index = sample.edge_index.numpy()
    x = sample.x.numpy()
    
    # 画边
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # 只画一次
            fig.add_trace(go.Scatter3d(
                x=[x[src, 0], x[dst, 0]],
                y=[x[src, 1], x[dst, 1]],
                z=[x[src, 2], x[dst, 2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
    
    # 画节点
    fig.add_trace(go.Scatter3d(
        x=x[:, 0], y=x[:, 1], z=x[:, 2],
        mode='markers',
        marker=dict(size=4, color=colors[type_id]),
        name=type_names[type_id]
    ))
    
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        height=500,
        title=f"{type_names[type_id]} - 样本 #{sample_idx}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== 页面2：训练结果 ====================
elif page == "📈 训练结果":
    st.title("📈 训练结果")
    
    # 显示损失曲线
    if os.path.exists("outputs/loss_curve.png"):
        st.subheader("训练损失曲线")
        st.image("outputs/loss_curve.png", width="stretch")
    else:
        st.warning("未找到损失曲线图片，请先运行 train.py 进行训练。")
    
    # 显示混淆矩阵
    if os.path.exists("outputs/confusion_matrix.png"):
        st.subheader("分类混淆矩阵")
        st.image("outputs/confusion_matrix.png", width="stretch")
    else:
        st.warning("未找到混淆矩阵图片，请先运行 train.py 进行训练。")
    
    # 模型信息
    st.subheader("模型信息")
    st.write(f"- 模型架构: 2层GCN (3→32→16)")
    st.write(f"- 对比损失: InfoNCE (温度τ=0.5)")
    st.write(f"- 数据增强: 节点丢弃 (丢弃率15%)")
    st.write(f"- 优化器: Adam (学习率0.01)")
    st.write(f"- 训练轮数: 30 epochs")
    total_params = sum(p.numel() for p in model.parameters())
    st.write(f"- 模型参数量: {total_params}")

# ==================== 页面3：嵌入可视化 ====================
elif page == "🔬 嵌入可视化":
    st.title("🔬 嵌入空间可视化")
    st.write("下图展示60个神经元在对比学习后的16维嵌入空间，经UMAP降维到3D后的分布。"
             "同类神经元应聚集在一起，不同类应彼此分离。")
    
    # UMAP降维
    try:
        import umap
        reducer = umap.UMAP(n_components=3, random_state=42)
        emb_3d = reducer.fit_transform(embeddings)
        
        fig = go.Figure()
        for label_id, name in type_names.items():
            mask = labels == label_id
            fig.add_trace(go.Scatter3d(
                x=emb_3d[mask, 0],
                y=emb_3d[mask, 1],
                z=emb_3d[mask, 2],
                mode='markers',
                marker=dict(size=6, color=colors[label_id]),
                name=name,
                text=[f"样本#{i}" for i in np.where(mask)[0]],
                hovertemplate='<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            scene=dict(xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3'),
            height=600,
            title="神经元嵌入空间3D可视化（UMAP降维）"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D版本
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        emb_2d = reducer_2d.fit_transform(embeddings)
        
        fig2 = go.Figure()
        for label_id, name in type_names.items():
            mask = labels == label_id
            fig2.add_trace(go.Scatter(
                x=emb_2d[mask, 0],
                y=emb_2d[mask, 1],
                mode='markers+text',
                marker=dict(size=10, color=colors[label_id]),
                name=name,
                text=[str(i) for i in np.where(mask)[0]],
                textposition='top center',
                textfont=dict(size=8)
            ))
        
        fig2.update_layout(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            height=500,
            title="神经元嵌入空间2D可视化"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    except ImportError:
        st.error("未安装 umap-learn，请运行: pip install umap-learn")

# ==================== 页面4：预测演示 ====================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一个预设的测试样本，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()
    st.subheader("💡 方式一：上传真实神经元切片（PNG/JPG）")
    uploaded_file = st.file_uploader("选择图片", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # 1. 图像预处理与特征提取
        from PIL import Image
        import cv2
        
        img = Image.open(uploaded_file).convert("L")  # 灰度化
        img_np = np.array(img)
        _, binary = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
        
        # 计算3个特征
        area = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])  # 面积占比
        edges = cv2.Canny(img_np, 100, 200)
        fractal_dim = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])  # 边缘复杂度
        h, w = np.where(binary > 0)
        if len(h) > 0:
            eccentricity = (max(h) - min(h)) / (max(w) - min(w) + 1e-5)  # 偏心率
        else:
            eccentricity = 0.5
            
        feat = torch.tensor([[area, fractal_dim, eccentricity]], dtype=torch.float)
        
        # 构造单节点图结构（模拟模型的输入格式）
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)
        
        # 2. 模型推理
        with torch.no_grad():
            pred_emb = model(feat, edge_index, batch).squeeze(0)
            
        # 3. 计算与训练集所有样本的相似度
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(pred_emb.numpy().reshape(1, -1), embeddings)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]
        
        # 4. 页面展示结果
        st.success("✅ 预测分析完成！")
        st.write(f"**提取的3维特征：** 面积占比 `{area:.3f}` | 边缘复杂度 `{fractal_dim:.3f}` | 偏心率 `{eccentricity:.3f}`")
        
           # --- 开始替换 ---
    st.write("*** 🏆 最相似的 5 个训练样本: ***")

    # 1. 先计算当前样本的嵌入向量 (Embedding)
    with torch.no_grad():
        # 注意：这里假设 sample 变量已经包含 .x 和 .edge_index
        emb = model(sample.x, sample.edge_index, torch.zeros(sample.x.size(0), dtype=torch.long))
        emb = emb.squeeze()

    # 2. 计算与所有训练集样本的余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    # emb 是 (D,), embeddings 是 (N, D) -> 结果 sims 是 (N,)
    sims = cosine_similarity(emb.numpy().reshape(1, -1), embeddings)[0]

    # 3. 获取相似度最高的 5 个索引 (从大到小排序)
    top5_idx = np.argsort(sims)[-5:][::-1]

    # 4. 循环展示结果
    for rank, idx in enumerate(top5_idx):
        # 【关键修复】：直接用 sims[idx] 获取原始相似度，不要乘以 100 除非你想显示百分比
        # idx 是原始数据集的下标，sims 是完整的相似度数组，这样取是安全的
        sim_val = sims[idx] * 100
        
        # 获取真实标签名称
        true_type = type_names[labels[idx]]
        
        # 打印每一行
        st.caption(
            f"第 {rank+1} 名: **样本#{idx}** | 真实类型: **{true_type}** | 相似度: **{sim_val:.1f}%**"
        )

    st.markdown("---")
    # --- 结束替换 ---

    st.subheader("🗄️ 方式二：选择预设测试样本")
    sample_options = {f"样本#{i} ({type_names[labels[i]]})": i for i in range(len(all_data))}
    selected = st.selectbox("选择测试样本", list(sample_options.keys()))
    sample_idx = sample_options[selected]
    sample = all_data[sample_idx]
    
    # 模型预测
  
    
    true_label = labels[sample_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("样本信息")
        st.write(f"- **样本编号**: #{sample_idx}")
        st.write(f"- **真实类型**: {type_names[true_label]}")
        st.write(f"- **节点数**: {sample.num_nodes}")
        st.write(f"- **边数**: {sample.num_edges}")
        
        st.subheader("最近邻样本（余弦相似度）")
        for rank, idx in enumerate(top5_idx):
            sim = sims[idx]
            match = "✅" if labels[idx] == true_label else "❌"
            st.write(f"{rank+1}. 样本#{idx} - {type_names[labels[idx]]} - 相似度: {sim:.4f} {match}")
    
    with col2:
        st.subheader("3D结构可视化")
        fig = go.Figure()
        edge_index = sample.edge_index.numpy()
        x = sample.x.numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:
                fig.add_trace(go.Scatter3d(
                    x=[x[src, 0], x[dst, 0]],
                    y=[x[src, 1], x[dst, 1]],
                    z=[x[src, 2], x[dst, 2]],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))
        
        fig.add_trace(go.Scatter3d(
            x=x[:, 0], y=x[:, 1], z=x[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors[true_label]),
            name=type_names[true_label]
        ))
        
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)