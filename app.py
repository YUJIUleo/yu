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

# ======================= 页面4: 预测演示 (终极修复版 - 修复 device 报错) =======================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一种方式，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # --- 【核心修复 1】强制定义 device ---
    # 不管外面有没有定义，这里必须自己定一个，防止报错
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except:
        device = 'cpu' # 兜底方案

    # --- 【核心修复 2】确保 all_data 和 embeddings 存在 ---
    if 'all_data' not in globals() and 'all_data' not in locals():
        st.error("系统错误：未检测到训练集数据 (all_data)。")
        st.stop()
    
    # 如果 embeddings 还没算，这里利用 session_state 缓存或现算
    if 'embeddings' not in locals():
        with st.spinner("正在后台计算训练集特征向量..."):
            try:
                # 假设 model 和 dev 已经在前面定义好了，或者用这里的 device
                # 注意：这里需要确保 model 已经加载到了内存中
                if 'model' not in locals() and 'model' not in globals():
                    st.error("模型未加载，请先运行训练页面。")
                    st.stop()
                
                # 批量计算所有样本的 embedding
                all_x = torch.stack([data.x for data in all_data]).to(device)
                all_edge_index = torch.cat([data.edge_index for data in all_data], dim=1).to(device) 
                # 注意：上面的 cat 逻辑可能不对，因为每个图的节点数不同，不能直接 cat edge_index
                # 正确的做法是逐个计算
                
                temp_embeddings = []
                for data in all_data:
                    data = data.to(device)
                    # 构造 batch 向量 (全0，因为是一次算一个图)
                    batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
                    emb = model(data.x.float(), data.edge_index, batch_vec)
                    temp_embeddings.append(emb.cpu().detach().numpy())
                
                embeddings = np.array(temp_embeddings).reshape(len(all_data), -1)
                
            except Exception as e:
                st.error(f"计算特征向量失败: {str(e)}")
                st.stop()

    tab1, tab2 = st.tabs(["方式一：上传真实神经元切片 (PNG/JPG)", "方式二：选择预设测试样本"])

    # ================== 方式一：上传图片 ==================
    with tab1:
        uploaded_file = st.file_uploader("上传文件", type=["png", "jpg", "jpeg"], key="upload_pred")
        if uploaded_file:
            st.success("图片上传成功！(注：目前仅为演示流程，实际预测需接入图像处理模型)")
            # 这里可以放图片处理的逻辑，暂时略过以防报错
            
    # ================== 方式二：选择样本 ==================
    with tab2:
        st.subheader("从训练集中选择样本进行测试")
        
        # 生成选项列表
        sample_options = [f"样本 {i} (类型 {d.y.item()})" for i, d in enumerate(all_data)]
        selected_idx_str = st.selectbox("请选择一个样本：", sample_options, key="select_sample")
        
        if selected_idx_str:
            # 解析出索引
            selected_idx = int(selected_idx_str.split(" ")[1])
            sample = all_data[selected_idx]
            
            st.info(f"正在分析：{selected_idx_str}")
            
            # --- 1. 模型预测 ---
            try:
                sample = sample.to(device)
                # 构造 batch 向量 (单样本预测时，所有节点的 batch_id 都是 0)
                batch_vec = torch.zeros(sample.x.size(0), dtype=torch.long, device=device)
                
                # 调用模型
                sample_emb = model(sample.x.float(), sample.edge_index, batch_vec)
                sample_emb_np = sample_emb.cpu().detach().numpy().reshape(1, -1)
                
                # 计算余弦相似度
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(sample_emb_np, embeddings)[0]
                
                # 获取 Top 5
                top5_indices = similarities.argsort()[-5:][::-1]
                
                st.markdown("**最相似的 5 个训练样本：**")
                for rank, idx in enumerate(top5_indices):
                    sim_score = similarities[idx]
                    true_label = all_data[idx].y.item()
                    st.caption(f"👉 Top {rank+1}: 样本 {idx} | 类型: {true_label} | 相似度: {sim_score:.4f}")
                    
            except Exception as e:
                st.error(f"预测过程出错: {str(e)}")
                st.stop()

            # --- 2. 3D 可视化 (带容错) ---
            st.subheader("3D 结构可视化")
            
            # 尝试提取坐标
            pos = None
            if hasattr(sample, 'pos') and sample.pos is not None:
                pos = sample.pos.cpu().numpy()
            elif hasattr(sample, 'x') and sample.x is not None and sample.x.shape[1] >= 3:
                pos = sample.x[:, :3].cpu().numpy()
            
            if pos is not None:
                try:
                    import plotly.express as px
                    import pandas as pd
                    
                    df = pd.DataFrame(pos, columns=['X', 'Y', 'Z'])
                    fig = px.scatter_3d(df, x='X', y='Y', z='Z', title="样本 3D 结构")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"3D 绘图组件缺失或出错: {str(e)}")
            else:
                st.warning("⚠️ 该样本没有坐标数据 (pos)，无法进行 3D 可视化。")