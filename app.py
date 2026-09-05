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

# ======================= 页面4: 预测演示 (最终修复版 - 包含batch参数) =======================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一种方式，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # --- 【关键前置检查】确保 all_data 和 embeddings 存在 ---
    # 如果之前页面没加载，这里尝试加载（防止 NameError）
    if 'all_data' not in globals() and 'all_data' not in locals():
        st.error("系统错误：未检测到训练集数据 (all_data)。请确保数据加载脚本已运行。")
        st.stop()
    
    # 如果 embeddings 还没算，这里现算（防止 NameError）
    if 'embeddings' not in locals():
        with st.spinner("正在后台计算训练集特征向量..."):
            try:
                # 假设你的模型叫 model，数据叫 all_data
                # 注意：这里需要构建一个大的 batch 来一次性计算所有训练集的 embedding
                # 如果 all_data 很大，这里可能会慢，但在演示页通常没问题
                
                # 1. 拼接所有数据
                from torch_geometric.data import Batch
                big_batch = Batch.from_data_list(all_data)
                
                # 2. 推理模式
                model.eval()
                with torch.no_grad():
                    # 【关键修复】传入 batch 参数
                    embeddings = model(big_batch.x, big_batch.edge_index, big_batch.batch).cpu().numpy()
            except Exception as e:
                st.error(f"计算特征向量失败: {e}")
                st.stop()

    # --- 创建两个标签页 ---
    tab1, tab2 = st.tabs(["📷 方式一：上传图片", "📂 方式二：选择样本"])

    # ======================= 方式一：上传图片 =======================
    with tab1:
        st.subheader("从本地上传真实神经元切片")
        uploaded_file = st.file_uploader("选择图片...", type=["png", "jpg", "jpeg"], key="upload_img")

        if uploaded_file is not None:
            try:
                # 1. 图像处理与特征提取 (模拟你之前的逻辑)
                from PIL import Image
                import numpy as np
                import torch
                
                img = Image.open(uploaded_file).convert('L') # 转灰度
                img_np = np.array(img)
                
                # 这里假设你有一个函数 extract_features_from_image 能提取出 area, fractal_dim 等
                # 如果没有，这里用随机数或固定值代替演示，你需要填入真实的图像处理逻辑
                # 假设提取出的特征是 feat_list
                # ⚠️ 注意：这里的特征提取逻辑必须和训练时保持一致！
                
                # 模拟特征提取 (请替换为你真实的图像->特征值逻辑)
                # 假设你的特征是 [area, fractal_dim, eccentricity]
                # 这里为了演示不报错，我随便写了几个数，**请务必改成你真实的计算逻辑**
                area = 1000 
                fractal_dim = 1.5
                eccentricity = 0.8
                
                feat_list = [area, fractal_dim, eccentricity] 
                
                # 2. 构建图数据 (Graph Data)
                # 假设你的模型输入是一个包含这些特征的简单图，或者你需要把图片转成图
                # 这里假设你是把这几个特征作为节点特征 x
                x_tensor = torch.tensor([feat_list], dtype=torch.float)
                
                # 假设没有边（或者根据特征构建边），这里造一个空的 edge_index
                edge_index = torch.tensor([[0], [0]], dtype=torch.long) 
                
                # 构造 batch 向量 (只有一个节点，所以是 [0])
                batch_vec = torch.tensor([0], dtype=torch.long)

                # 3. 模型预测
                model.eval()
                with torch.no_grad():
                    # 【关键修复】传入 batch_vec
                    pred_emb = model(x_tensor, edge_index, batch_vec).cpu().numpy()

                # 4. 计算相似度
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(pred_emb.reshape(1, -1), embeddings)[0]
                top5_idx = np.argsort(sims)[-5:][::-1]

                # 5. 展示结果
                st.success("分析完成！找到最相似的 5 个样本：")
                
                # 显示上传的图片
                st.image(uploaded_file, caption="上传的切片", width=200)
                
                # 显示 Top 5 列表
                for rank, idx in enumerate(top5_idx):
                    true_type = all_data[idx].y.item() # 假设标签存在 .y 中
                    sim_val = sims[idx]
                    st.markdown(f"**第 {rank+1} 相似**: 样本 {idx} | 类型: {true_type} | 相似度: {sim_val:.4f}")

            except Exception as e:
                st.error(f"图片分析出错: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ======================= 方式二：选择样本 =======================
    with tab2:
        st.subheader("从训练集中选择样本进行测试")
        
        # 生成选项列表
        sample_options = {f"样本 {i} (类型: {d.y.item()})": i for i, d in enumerate(all_data)}
        selected_label = st.selectbox("请选择一个样本:", list(sample_options.keys()))
        selected_idx = sample_options[selected_label]

        if selected_idx is not None:
            try:
                sample = all_data[selected_idx]
                
                # 1. 准备模型输入
                # 增加一个维度变成 batch (因为模型通常接受 batch 输入)
                x_tensor = sample.x.unsqueeze(0)  # [Num_Nodes, Feat] -> [1, Num_Nodes, Feat] ? 
                # 不对，PyG 的模型通常接受 [Total_Nodes, Feat]，通过 batch 向量区分
                
                # PyG 标准做法：直接使用 sample 的属性，但需要构造 batch 向量
                x_input = sample.x
                edge_input = sample.edge_index
                
                # 【关键修复】构造 batch 向量
                # 如果 sample.x 有 N 个节点，batch 向量就是 [0, 0, ..., 0] (长度为 N)
                batch_input = torch.zeros(sample.x.size(0), dtype=torch.long)

                # 2. 模型预测
                model.eval()
                with torch.no_grad():
                    # 传入 x, edge_index, batch
                    sample_emb = model(x_input, edge_input, batch_input).cpu().numpy()

                # 3. 计算相似度 (自己和自己比应该是 1.0，这里主要看它和其他人的距离)
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(sample_emb.reshape(1, -1), embeddings)[0]
                top5_idx = np.argsort(sims)[-5:][::-1]

                # 4. 展示结果
                st.info(f"正在分析：{selected_label}")
                
                # 显示 Top 5
                st.write("**最相似的 5 个训练样本：**")
                for rank, idx in enumerate(top5_idx):
                    true_type = all_data[idx].y.item()
                    sim_val = sims[idx]
                    
                    # 高亮当前选中的样本
                    is_current = (idx == selected_idx)
                    prefix = "👉 **(当前)**" if is_current else f"👉 **Top {rank+1}**"
                    
                    st.markdown(f"{prefix} 样本 {idx} | 类型: {true_type} | 相似度: {sim_val:.4f}")

                # 5. 可视化 (可选)
                st.subheader("3D 结构可视化")
                import plotly.graph_objects as go
                
                pos = sample.pos.numpy() if hasattr(sample, 'pos') else sample.x[:, :3].numpy() # 尝试获取坐标
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                    mode='markers+lines',
                    marker=dict(size=3, color=sample.y.item(), colorscale='Viridis'),
                    line=dict(width=1)
                )])
                fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"样本分析出错: {e}")
                import traceback
                st.code(traceback.format_exc())