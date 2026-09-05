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

# ======================= 页面4: 预测演示 (修复版) =======================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一种方式，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # --- 【核心修复】在这里强制确保 embeddings 存在 ---
    # 使用 session_state 来缓存数据，防止每次操作都重算，也防止跨页面丢失
    if 'cached_embeddings' not in st.session_state or 'cached_all_data' not in st.session_state:
        with st.spinner("正在后台加载训练集数据并计算特征向量...请稍候..."):
            try:
                # 1. 加载数据 (假设 all_data 是你的数据集列表)
                # 如果 all_data 是全局导入的，直接用；如果是文件，这里要 load
                # 这里假设你能访问到 all_data，如果不能，请取消下面注释并修改路径
                # from your_data_module import load_data 
                # all_data = load_data() 
                
                # 2. 计算 embeddings
                model.eval()
                all_x = torch.stack([d.x for d in all_data])
                all_edge_index = torch.cat([d.edge_index for d in all_data], dim=1)
                # 注意：如果数据量大，建议分批(batch)推理，这里假设数据量能一次跑完
                # 如果显存不够，请改用循环 batch 推理
                
                with torch.no_grad():
                    # 这里的 model 必须是你训练好的那个模型实例
                    embs = model(all_x, all_edge_index) 
                
                # 3. 存入 session_state
                st.session_state['cached_embeddings'] = embs.cpu().numpy()
                st.session_state['cached_all_data'] = all_data
                
            except Exception as e:
                st.error(f"数据加载失败: {e}")
                st.stop()

    # 从缓存中读取数据
    embeddings = st.session_state['cached_embeddings']
    all_data = st.session_state['cached_all_data']
    
    # --- 辅助函数：展示 Top 5 结果 ---
    def show_top5_results(pred_emb_np):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        sims = cosine_similarity(pred_emb_np.reshape(1, -1), embeddings)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]
        
        st.success("✅ 预测分析完成！")
        
        # 显示匹配列表
        st.markdown("---")
        st.subheader("🏆 最相似的 5 个训练样本:")
        
        cols = st.columns([1, 2, 2, 2])
        cols[0].write("**排名**")
        cols[1].write("**样本ID/类型**")
        cols[2].write("**真实类型**")
        cols[3].write("**相似度**")
        
        for rank, idx in enumerate(top5_idx):
            sim_val = sims[idx] * 100
            true_type = type_names[all_data[idx].y.item()] # 假设 type_names 已定义
            
            c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
            c1.write(f"#{rank+1}")
            c2.write(f"样本 #{idx}")
            c3.write(true_type)
            c4.metric(label="", value=f"{sim_val:.2f}%")
            
        # 显示第一个最相似样本的图 (可选)
        st.markdown("---")
        st.subheader("📊 最佳匹配样本可视化:")
        best_sample = all_data[top5_idx[0]]
        # ...这里放你之前的绘图代码 (plotly_chart)...

    # ==========================================
    # 方式一：上传图片
    # ==========================================
    st.header("📷 方式一：上传真实神经元切片 (PNG/JPG)")
    uploaded_file = st.file_uploader("选择图片...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            from PIL import Image
            import cv2
            
            img = Image.open(uploaded_file).convert("L")
            img_np = np.array(img)
            _, binary = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
            
            # 计算特征 (复用你之前的逻辑)
            area = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
            edges = cv2.Canny(binary, 100, 200)
            fractal_dim = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            h, w = np.where(binary > 0)
            if len(h) > 0:
                eccentricity = (max(h) - min(h)) / (max(w) - min(w) + 1e-5)
            else:
                eccentricity = 0.5
                
            feat = torch.tensor([[area, fractal_dim, eccentricity]], dtype=torch.float)
            
            # 构造图数据 (这里假设你需要构造一个临时的图来跑模型)
            # 注意：这里需要和你训练时的输入格式一致。
            # 如果你的模型直接吃 feature 向量，就不需要 edge_index。
            # 看你的截图，你构造了 edge_index，说明模型是 GNN。
            # 但 GNN 通常需要节点特征。这里假设你是把整张图作为一个节点？
            # 或者是全连接图？
            
            # 模拟构造一个单节点或简单图结构 (根据你的模型需求调整)
            # 假设：单节点图，特征就是上面算的 area, fractal, ecc
            x_input = feat 
            edge_index_input = torch.tensor([[0],[0]], dtype=torch.long) 
            
            model.eval()
            with torch.no_grad():
                pred_emb = model(x_input, edge_index_input).squeeze(0)
                
            show_top5_results(pred_emb.numpy())
            
        except Exception as e:
            st.error(f"图片处理出错: {e}")

    # ==========================================
    # 方式二：选择预设样本
    # ==========================================
    st.header("📂 方式二：选择预设测试样本")
    
    # 生成选项
    sample_options = {f"样本#{i} ({type_names[d.y.item()]})": i for i, d in enumerate(all_data)}
    selected_label = st.selectbox("选择测试样本", list(sample_options.keys()))
    
    if selected_label:
        sample_idx = sample_options[selected_label]
        sample = all_data[sample_idx]
        
        model.eval()
        with torch.no_grad():
            # 直接拿 sample 里的数据跑
            sample_emb = model(sample.x, sample.edge_index).squeeze(0)
            
        show_top5_results(sample_emb.numpy())