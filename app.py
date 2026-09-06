"""
第三步：Streamlit 展示平台
运行方式：streamlit run app.py

修改说明（v2 - GCN/LCN预处理 + 特征扩充）：
  1. 新增全局对比度归一化(GCN)和局部对比度归一化(LCN)预处理
  2. 特征维度从 3 维扩充到 10 维（3个基础形态特征 + 7个Hu不变矩）
  3. 模型 in_dim 从 3 改为 10，第二层权重保留预训练知识
  4. 训练数据节点特征补齐到 10 维（前3维为原始坐标，后7维补0）
  5. 上传页面增加预处理四联图（原图 / GCN / LCN / 二值化）
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import os

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 页面配置 ====================
st.set_page_config(page_title="神经元GCN对比学习平台", page_icon="\U0001f9e0", layout="wide")

# ==================== GCN / LCN 预处理函数 ====================
def gcn_normalize(img, eps=1e-5):
    """全局对比度归一化 (Global Contrast Normalization)
    消除整张图像光照、曝光不一致的影响，使对比度统一。
    """
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img) + eps
    return (img - mean) / std


def lcn_normalize(img, window_size=7, eps=1e-5):
    """局部对比度归一化 (Local Contrast Normalization)
    消除局部光照变化的影响，增强纹理细节。
    """
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float32)
    local_mean = uniform_filter(img, size=window_size)
    std = np.std(img) + eps
    return (img - local_mean) / std


# ==================== 10维特征提取 ====================
def extract_10d_features(img_np):
    """从灰度图像中提取 10 维特征：
      - 3 个基础形态特征：面积占比、边缘密度、偏心率
      - 7 个 Hu 不变矩（对旋转、缩放、平移不变）
    所有特征在 GCN 预处理后的图像上计算，以提升对比度敏感性。
    """
    # 1. GCN 预处理（增强全局对比度）
    img_gcn = gcn_normalize(img_np)

    # 2. 二值化（Otsu 自动阈值）
    from cv2 import threshold, THRESH_BINARY_INV, THRESH_OTSU
    _, binary = threshold(img_gcn, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)

    # 3. 边缘检测
    from cv2 import Canny
    edges = Canny(img_gcn, 100, 200)

    # --- 基础形态特征 (3维) ---
    area_ratio = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    ys, xs = np.where(binary > 0)
    if len(ys) > 0:
        eccentricity = float(max(ys) - min(ys)) / (float(max(xs) - min(xs)) + 1e-5)
    else:
        eccentricity = 0.5

    # --- Hu 不变矩 (7维) ---
    from cv2 import moments, HuMoments
    raw_moments = moments(binary)
    hu = HuMoments(raw_moments).flatten()
    # Hu 矩数值范围差异大，取 log 压缩
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-6)

    # 拼接
    features = np.array([area_ratio, edge_density, eccentricity] + hu_log.tolist(), dtype=np.float32)
    return features


# ==================== 模型定义 ====================
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
    # 模型 in_dim 改为 10（对应 10 维特征）
    model = GCNEncoder(in_dim=10, hidden_dim=32, out_dim=16)

    if os.path.exists("models/gcn_encoder.pth"):
        pretrained = torch.load("models/gcn_encoder.pth", map_location="cpu", weights_only=True)

        # 复制第二层权重（conv2: 32→16，不受 in_dim 影响）
        conv2_state = {k.replace("conv2.", ""): v for k, v in pretrained.items() if "conv2" in k}
        model.conv2.load_state_dict(conv2_state, strict=False)

        # 第一层 (conv1): 原始 in_dim=3 → 新 in_dim=10
        # 前 3 列复制预训练权重，后 7 列保持随机初始化
        conv1_state = {k.replace("conv1.", ""): v for k, v in pretrained.items() if "conv1" in k}
        if "weight" in conv1_state:
            # conv1_state['weight'] shape: [32, 3]
            # model.conv1.weight shape: [32, 10]
            with torch.no_grad():
                model.conv1.weight[:, :3] = conv1_state["weight"]
                # 后 7 列保持随机初始化（Xavier uniform）
                nn.init.xavier_uniform_(model.conv1.weight[:, 3:])
        if "bias" in conv1_state:
            model.conv1.bias = conv1_state["bias"]

    model.eval()
    return model


all_data = load_data()
model = load_model()

# ==================== 预计算 UMAP 3D 坐标 ====================
@st.cache_data
def get_all_embeddings():
    """提取所有训练样本的嵌入，节点特征补齐到 10 维（前3维坐标 + 后7维0）"""
    with torch.no_grad():
        padded_data = []
        for d in all_data:
            # 补齐节点特征: [N, 3] → [N, 10]
            n_nodes = d.x.shape[0]
            padded_x = torch.cat([d.x, torch.zeros(n_nodes, 7)], dim=-1)
            padded_data.append(Data(x=padded_x, edge_index=d.edge_index, y=d.y))
        loader = Batch.from_data_list(padded_data)
        emb = model(loader.x, loader.edge_index, loader.batch)
    return emb.numpy()


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


embeddings = get_all_embeddings()
labels = np.array([d.y.item() for d in all_data])
type_names = {0: "HIP(锥体神经元)", 1: "DG(颗粒细胞)", 2: "CB(浦肯野细胞)"}
colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

umap_3d = get_umap_3d()

# ==================== 侧边栏导航 ====================
st.sidebar.title("\U0001f9e0 神经元对比学习平台")
page = st.sidebar.radio("选择页面", ["\U0001f4ca 数据概览", "\U0001f4c8 训练结果", "\U0001f52c 嵌入可视化", "\U0001f52e 预测演示"])

# ==================== 页面1：数据概览 ====================
if page == "\U0001f4ca 数据概览":
    st.title("\U0001f4ca 数据概览")

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
    st.write(f"- 节点特征维度: **10** (3维坐标 + 7维Hu矩补齐)")
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
elif page == "\U0001f4c8 训练结果":
    st.title("\U0001f4c8 训练结果")

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
    st.write(f"- 模型架构: 2层GCN (10→32→16)")
    st.write(f"- 对比损失: InfoNCE (温度τ=0.5)")
    st.write(f"- 数据增强: 节点丢弃 (丢弃率15%)")
    st.write(f"- 优化器: Adam (学习率0.01)")
    st.write(f"- 训练轮数: 30 epochs")
    total_params = sum(p.numel() for p in model.parameters())
    st.write(f"- 模型参数量: {total_params}")

# ==================== 页面3：嵌入可视化 ====================
elif page == "\U0001f52c 嵌入可视化":
    st.title("\U0001f52c 嵌入空间可视化")
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
elif page == "\U0001f52e 预测演示":
    st.title("\U0001f52e 新样本预测演示")
    st.write("选择一个预设的测试样本，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # ==================== 方式一：上传真实神经元切片 ====================
    st.subheader("\U0001f4a1 方式一：上传真实神经元切片（PNG/JPG）")
    st.info(
        "本模式对上传图像依次执行：GCN全局对比度归一化 → LCN局部对比度归一化 → "
        "Otsu二值化 → 提取10维特征（3个基础形态特征 + 7个Hu不变矩）→ 1节点图 → GCN推理 → 最近邻投票"
    )
    uploaded_file = st.file_uploader("选择图片", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        from PIL import Image
        import cv2

        # 1. 读取图像
        img_pil = Image.open(uploaded_file)
        img_rgb = np.array(img_pil)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 2. 预处理
        img_gcn = gcn_normalize(img_gray)
        img_lcn = lcn_normalize(img_gray)

        # 3. 二值化（Otsu）
        _, binary = cv2.threshold(img_gcn, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. 可视化预处理流水线
        fig_pre, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_gray, cmap='gray')
        axes[0].set_title("原始灰度图")
        axes[0].axis('off')

        axes[1].imshow(img_gcn, cmap='gray')
        axes[1].set_title("GCN 全局对比度归一化")
        axes[1].axis('off')

        axes[2].imshow(img_lcn, cmap='gray')
        axes[2].set_title("LCN 局部对比度归一化")
        axes[2].axis('off')

        axes[3].imshow(binary, cmap='gray')
        axes[3].set_title("Otsu 二值化")
        axes[3].axis('off')

        plt.tight_layout()
        st.pyplot(fig_pre)

        # 5. 提取 10 维特征
        features = extract_10d_features(img_gray)

        st.success("\u2705 预处理完成，特征提取成功！")
        feat_names = [
            "面积占比", "边缘密度", "偏心率",
            "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"
        ]
        for name, val in zip(feat_names, features):
            st.write(f"  - **{name}**: `{val:.4f}`")

        # 6. 构建 1 节点图并推理
        feat = torch.tensor([features], dtype=torch.float)  # [1, 10]
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            pred_emb = model(feat, edge_index, batch).squeeze(0)

        # 7. 最近邻搜索
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(pred_emb.numpy().reshape(1, -1), embeddings)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]

        # 8. 投票法预测
        vote = {}
        for idx in top5_idx:
            label = labels[idx]
            vote[label] = vote.get(label, 0) + 1
        pred_label = max(vote, key=vote.get)
        pred_confidence = vote[pred_label] / len(top5_idx)

        st.write(f"**预测结果**：{type_names[pred_label]}（置信度：{pred_confidence:.0%}）")

        st.write("**\U0001f3c6 最相似的 5 个训练样本：**")
        for i, idx in enumerate(top5_idx):
            sim_val = sims[idx] * 100
            true_type = type_names[labels[idx]]
            st.progress(sim_val / 100)
            st.caption(f"第{i+1}名：样本#{idx} | 真实类型：**{true_type}** | 相似度：**{sim_val:.1f}%**")

        # 9. 嵌入空间可视化
        st.subheader("\U0001f4ca 嵌入空间位置")

        fig = go.Figure()

        # 画背景云团
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

        # 画当前上传样本
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
    st.subheader("\U0001f5c4\ufe0f 方式二：选择预设测试样本")
    sample_options = {f"样本#{i} ({type_names[labels[i]]})": i for i in range(len(all_data))}
    selected = st.selectbox("选择测试样本", list(sample_options.keys()))
    sample_idx = sample_options[selected]
    sample = all_data[sample_idx]

    # 补齐节点特征到 10 维
    n_nodes = sample.x.shape[0]
    padded_x = torch.cat([sample.x, torch.zeros(n_nodes, 7)], dim=-1)

    # 模型推理
    with torch.no_grad():
        emb = model(padded_x, sample.edge_index, torch.zeros(n_nodes, dtype=torch.long))
        emb = emb.squeeze()

    # 最近邻搜索
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(emb.numpy().reshape(1, -1), embeddings)[0]
    top5_idx = np.argsort(sims)[-5:][::-1]

    true_label = labels[sample_idx]

    # 投票法预测
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
            match = "\u2705" if labels[idx] == true_label else "\u274c"
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

    # 嵌入空间3D可视化
    st.subheader("\U0001f4ca 嵌入空间分布（UMAP降维）")
    st.write("下图展示所有训练样本在嵌入空间的分布（按类型着色），**红色大球**代表你选择的当前样本。")

    fig2 = go.Figure()

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

    # 最终判定
    st.divider()
    st.success(f"\U0001f3af **最终判定：该样本属于 {type_names[pred_label]}**")
    st.info(
        f"""
        **分析总结：**
        - 通过最近邻投票法（Top-5 余弦相似度），{vote[pred_label]} 个邻居属于 **{type_names[pred_label]}**，预测置信度为 **{pred_confidence:.0%}**。
        - 图中 **红色大球** 代表你选择的样本，它在嵌入空间中落在 **{type_names[pred_label]}** 的聚集区域内。
        - 如果真实标签与预测一致，说明模型在对比学习中成功成功将该神经元分类到了正确的类型。
        """
    )