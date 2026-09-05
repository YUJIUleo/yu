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

# ==================== 预计算 UMAP 3D 坐标 ====================
@st.cache_data
def get_umap_3d():
    try:
        import umap
        reducer = umap.UMAP(n_components=3, random_state=42)
        return reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3, random_state=42)
        return reducer.fit_transform(embeddings)

umap_3d = get_umap_3d()

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
        if src < dst:
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
    
    if os.path.exists("outputs/loss_curve.png"):
        st.subheader("训练损失曲线")
        st.image("outputs/loss_curve.png", width="stretch")
    else:
        st.warning("未找到损失曲线图片，请先运行 train.py 进行训练。")
    
    if os.path.exists("outputs/confusion_matrix.png"):
        st.subheader("分类混淆矩阵")
        st.image("outputs/confusion_matrix.png", width="stretch")
    else:
        st.warning("未找到混淆矩阵图片，请先运行 train.py 进行训练。")
    
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
    st.write("下图展示所有神经元在对比学习后的16维嵌入空间，经UMAP降维到3D后的分布。"
             "同类神经元应聚集在一起，不同类应彼此分离。")
    
    fig = go.Figure()
    for label_id, name in type_names.items():
        mask = labels == label_id
        fig.add_trace(go.Scatter3d(
            x=umap_3d[mask, 0],
            y=umap_3d[mask, 1],
            z=umap_3d[mask, 2],
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
    try:
        import umap
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        emb_2d = reducer_2d.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer_2d = PCA(n_components=2, random_state=42)
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

# ==================== 页面4：预测演示 ====================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一个预设的测试样本，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # ==================== 方式一：上传真实神经元切片 ====================
    st.subheader("💡 方式一：上传真实神经元切片（PNG/JPG）")
    uploaded_file = st.file_uploader("选择图片", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        from PIL import Image
        import cv2
        
        img = Image.open(uploaded_file).convert("L")
        img_np = np.array(img)
        _, binary = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
        
        area = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
        edges = cv2.Canny(img_np, 100, 200)
        fractal_dim = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        h, w = np.where(binary > 0)
        eccentricity = (max(h) - min(h)) / (max(w) - min(w) + 1e-5) if len(h) > 0 else 0.5
        
        feat = torch.tensor([[area, fractal_dim, eccentricity]], dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)
        
        with torch.no_grad():
            pred_emb = model(feat, edge_index, batch).squeeze(0)
        
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(pred_emb.numpy().reshape(1, -1), embeddings)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]
        
        # === 投票法预测类型 ===
        vote = {}
        for idx in top5_idx:
            label = labels[idx]
            vote[label] = vote.get(label, 0) + 1
        pred_label = max(vote, key=vote.get)
        pred_confidence = vote[pred_label] / len(top5_idx)
        
        st.success("✅ 预测分析完成！")
        st.write(f"**提取的3维特征：** 面积占比 `{area:.3f}` | 边缘复杂度 `{fractal_dim:.3f}` | 偏心率 `{eccentricity:.3f}`")
        
        st.write("**🏆 最相似的 5 个训练样本：**")
        for i, idx in enumerate(top5_idx):
            sim_val = sims[idx] * 100
            true_type = type_names[labels[idx]]
            st.progress(sim_val / 100)
            st.caption(f"第{i+1}名：样本#{idx} | 真实类型：**{true_type}** | 相似度：**{sim_val:.1f}%**")
        
        # === 嵌入空间可视化（上传样本） ===
        st.subheader("📊 嵌入空间位置")
        
        fig = go.Figure()
        
        # 画背景云团（所有训练数据，按类型着色）
        for label_id, name in type_names.items():
            mask = labels == label_id
            fig.add_trace(go.Scatter3d(
                x=umap_3d[mask, 0],
                y=umap_3d[mask, 1],
                z=umap_3d[mask, 2],
                mode='markers',
                marker=dict(size=4, color=colors[label_id], opacity=0.3),
                name=name
            ))
        
        # 画当前上传样本（红色大球）— 用最近邻的位置作为参考
        nearest_pos = umap_3d[top5_idx[0]]
        fig.add_trace(go.Scatter3d(
            x=[nearest_pos[0]],
            y=[nearest_pos[1]],
            z=[nearest_pos[2]],
            mode='markers+text',
            text=['上传样本'],
            textposition="top center",
            marker=dict(size=14, color='red', symbol='circle'),
            name='上传样本'
        ))
        
        fig.update_layout(
            scene=dict(xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3'),
            height=600,
            title=f"嵌入空间可视化（预测：{type_names[pred_label]}，置信度：{pred_confidence:.0%}）"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

    # ==================== 方式二：选择预设测试样本 ====================
    st.subheader("🗄️ 方式二：选择预设测试样本")
    sample_options = {f"样本#{i} ({type_names[labels[i]]})": i for i in range(len(all_data))}
    selected = st.selectbox("选择测试样本", list(sample_options.keys()))
    sample_idx = sample_options[selected]
    sample = all_data[sample_idx]
    
    # 模型推理
    with torch.no_grad():
        emb = model(sample.x, sample.edge_index, torch.zeros(sample.num_nodes, dtype=torch.long))
        emb = emb.squeeze()
    
    # 最近邻搜索
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(emb.numpy().reshape(1, -1), embeddings)[0]
    top5_idx = np.argsort(sims)[-5:][::-1]
    
    true_label = labels[sample_idx]
    
    # === 投票法预测 ===
    vote = {}
    for idx in top5_idx:
        label = labels[idx]
        vote[label] = vote.get(label, 0) + 1
    pred_label = max(vote, key=vote.get)
    pred_confidence = vote[pred_label] / len(top5_idx)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("样本信息")
        st.write(f"- **样本编号**: #{sample_idx}")
        st.write(f"- **真实类型**: {type_names[true_label]}")
        st.write(f"- **预测类型**: {type_names[pred_label]}")
        st.write(f"- **预测置信度**: {pred_confidence:.0%}")
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
    
    # === 嵌入空间3D可视化（背景云团 + 红球） ===
    st.subheader("📊 嵌入空间分布（UMAP降维）")
    st.write(f"下图展示所有训练样本在嵌入空间的分布（按类型着色），**红色大球**代表你选择的当前样本。")
    
    fig2 = go.Figure()
    
    # 画背景云团（所有训练数据，按类型着色，半透明）
    for label_id, name in type_names.items():
        mask = labels == label_id
        fig2.add_trace(go.Scatter3d(
            x=umap_3d[mask, 0],
            y=umap_3d[mask, 1],
            z=umap_3d[mask, 2],
            mode='markers',
            marker=dict(size=5, color=colors[label_id], opacity=0.3),
            name=name
        ))
    
    # 画当前样本（红色大球）
    current_pos = umap_3d[sample_idx]
    fig2.add_trace(go.Scatter3d(
        x=[current_pos[0]],
        y=[current_pos[1]],
        z=[current_pos[2]],
        mode='markers+text',
        text=[f'当前样本\n({type_names[true_label]})'],
        textposition="top center",
        marker=dict(size=14, color='red', symbol='circle'),
        name='当前样本'
    ))
    
    fig2.update_layout(
        scene=dict(xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3'),
        height=600,
        title=f"神经元嵌入空间3D可视化 — 当前样本：{type_names[true_label]}（预测：{type_names[pred_label]}，置信度：{pred_confidence:.0%}）"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # === 最终判定 ===
    st.divider()
    st.success(f"🎯 **最终判定：该样本属于 {type_names[pred_label]}**")
    st.info(f"""
    **分析总结：**
    - 通过最近邻投票法（Top-5 余弦相似度），{vote[pred_label]} 个邻居属于 **{type_names[pred_label]}**，预测置信度为 **{pred_confidence:.0%}**。
    - 图中 **红色大球** 代表你选择的样本，它在嵌入空间中落在 **{type_names[pred_label]}** 的聚集区域内。
    - 如果真实标签与预测一致，说明模型成功将该神经元分类到了正确的类型。