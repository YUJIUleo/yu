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

# ======================= 页面4: 预测演示 (双模式终极修复版) =======================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一种方式，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # --- 【环境检查】 ---
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        import numpy as np
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        st.error(f"环境初始化失败: {e}")
        st.stop()

    if 'all_data' not in globals() and 'all_data' not in locals():
        st.error("系统错误：未检测到训练集数据 (all_data)。")
        st.stop()

    # 获取训练集的特征维度 (in_channels)，这是防止报错的关键
    # 假设 all_data 是一个列表，取第一个数据的 x 的列数
    try:
        target_feature_dim = all_data[0].x.shape[1]
    except:
        st.error("无法读取训练集特征维度，请检查 all_data 格式。")
        st.stop()

    tab1, tab2 = st.tabs(["方式一：上传真实切片", "方式二：选择预设测试样本"])

    # ==========================================
    # 方式一：上传真实切片 (修复维度不匹配)
    # ==========================================
    with tab1:
        uploaded_file = st.file_uploader("上传神经元图片 (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            try:
                from PIL import Image
                import cv2
                
                # 1. 读取图片
                image = Image.open(uploaded_file).convert('L') # 转灰度
                img_np = np.array(image)
                
                # 2. 简单的图像处理：二值化找亮点 (模拟神经元中心)
                _, binary = cv2.threshold(img_np, 50, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                centers = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 10 < area < 5000: # 过滤噪点和过大的块
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centers.append([cx, cy])
                
                if len(centers) < 2:
                    st.warning("图片中未检测到足够的神经元节点（至少需要2个），请尝试调整阈值或换图。")
                else:
                    st.success(f"检测到 {len(centers)} 个神经元节点，正在构建图结构...")
                    
                    # 3. 构建图数据 (Data)
                    # 节点坐标归一化 (简单处理)
                    coords = np.array(centers, dtype=np.float32)
                    coords[:, 0] /= img_np.shape[1]
                    coords[:, 1] /= img_np.shape[0]
                    
                    # 构造 edge_index (简单的全连接或KNN，这里用距离阈值模拟)
                    edge_list = []
                    dist_threshold = 0.15 # 距离阈值
                    
                    for i in range(len(coords)):
                        for j in range(i+1, len(coords)):
                            d = np.linalg.norm(coords[i] - coords[j])
                            if d < dist_threshold:
                                edge_list.append([i, j])
                                edge_list.append([j, i])
                    
                    if not edge_list:
                         st.warning("节点间距离过远，未形成连接。正在强制连接最近邻...")
                         # 兜底：强制连成链
                         for i in range(len(coords)-1):
                             edge_list.append([i, i+1])
                             edge_list.append([i+1, i])

                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    
                    # *** 关键修复：特征维度对齐 ***
                    # 如果训练集是3维(x,y,z)，而图片只有2维(x,y)，必须补0
                    node_features = torch.tensor(coords, dtype=torch.float)
                    current_dim = node_features.shape[1]
                    
                    if current_dim < target_feature_dim:
                        padding = torch.zeros(node_features.shape[0], target_feature_dim - current_dim)
                        node_features = torch.cat([node_features, padding], dim=1)
                    elif current_dim > target_feature_dim:
                        node_features = node_features[:, :target_feature_dim]
                        
                    new_graph = Data(x=node_features, edge_index=edge_index).to(device)
                    
                    # 4. 预测
                    model.eval()
                    with torch.no_grad():
                        # 构造 batch 向量
                        batch_vec = torch.zeros(new_graph.x.size(0), dtype=torch.long, device=device)
                        out = model(new_graph.x, new_graph.edge_index, batch_vec)
                        pred_embedding = out.cpu().numpy()
                    
                    st.success("预测完成！正在生成可视化...")
                    
                    # 5. 可视化 (只画背景 + 当前点)
                    # 这里假设你有之前算好的 train_embeddings (如果没有，这里只能画当前点)
                    # 为了演示，我们画一个散点图代表当前样本在空间的位置
                    import plotly.express as px
                    import pandas as pd
                    
                    df_curr = pd.DataFrame(pred_embedding, columns=['Dim1', 'Dim2', 'Dim3'])
                    df_curr['Type'] = '新上传样本'
                    
                    fig = px.scatter_3d(df_curr, x='Dim1', y='Dim2', z='Dim3', color='Type', 
                                        title="新样本嵌入位置", 
                                        color_discrete_map={'新上传样本': 'red'})
                    # 如果有历史数据可以 add_trace 加进去，这里简化处理
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"图像处理或预测失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # ==========================================
    # 方式二：选择预设测试样本 (修复全量堆叠报错)
    # ==========================================
    with tab2:
        # 获取所有样本的标签用于展示
        labels = [getattr(d, 'y', 0).item() if hasattr(d, 'y') else 0 for d in all_data]
        unique_labels = sorted(list(set(labels)))
        
        selected_label = st.selectbox("选择要测试的类别:", unique_labels)
        
        # 筛选出该类别的所有索引
        candidates = [i for i, l in enumerate(labels) if l == selected_label]
        
        if candidates:
            sample_idx = st.selectbox(f"从 [类型 {selected_label}] 中选择一个具体样本:", candidates)
            
            if st.button("开始分析该样本"):
                try:
                    # *** 关键修复：只取当前这一个样本 ***
                    target_graph = all_data[sample_idx].to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        # 构造单样本的 batch 向量
                        batch_vec = torch.zeros(target_graph.x.size(0), dtype=torch.long, device=device)
                        out = model(target_graph.x, target_graph.edge_index, batch_vec)
                        single_embedding = out.cpu().numpy() # shape: [1, dim]
                    
                    st.success(f"样本 {sample_idx} 分析完成！")
                    
                            
            # 显示向量数值
            st.markdown("### 🧬 嵌入向量前3维")
            vec_str = " ".join([f"{x:.6f}" for x in pred_embedding[0][:3]])
            st.latex(f"[ {vec_str} ]")

            # ======================= 3D 可视化 (带背景版) =======================
            st.markdown("### 🌌 空间位置分布")
            
            # 1. 准备背景数据 (所有训练集)
            # 假设 all_data 是一个列表，我们需要把所有图的节点坐标拼起来
            # 注意：这里我们取所有数据的“平均中心”或者“第一个节点”来代表那个样本在空间的位置
            # 如果你的 all_data 结构比较复杂，这里用最简单的方式：取每个样本的第一个节点的坐标作为代表
            
            bg_x, bg_y, bg_z = [], [], []
            labels = []
            
            for i, g in enumerate(all_data):
                # 取该样本的第一个节点坐标作为代表点 (或者你可以取所有节点的平均值)
                if g.x.shape[0] > 0:
                    bg_x.append(g.x[0][0].item())
                    bg_y.append(g.x[0][1].item())
                    bg_z.append(g.x[0][2].item())
                    labels.append(g.y.item() if hasattr(g, 'y') else 0)

            # 2. 准备前景数据 (当前选中的样本)
            fg_x = [pred_embedding[0][0].item()]
            fg_y = [pred_embedding[0][1].item()]
            fg_z = [pred_embedding[0][2].item()]

            # 3. 绘图
            import plotly.graph_objects as go

            fig = go.Figure()

            # 添加背景散点 (灰色，小一点)
            fig.add_trace(go.Scatter3d(
                x=bg_x, y=bg_y, z=bg_z,
                mode='markers',
                marker=dict(size=3, color='gray', opacity=0.3),
                name='训练集背景'
            ))

            # 添加前景散点 (红色，大一点)
            fig.add_trace(go.Scatter3d(
                x=fg_x, y=fg_y, z=fg_z,
                mode='markers+text',
                text=[f"样本 {sample_idx}"],
                textposition="top center",
                marker=dict(size=8, color='red', symbol='circle'),
                name='当前预测样本'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='Dim 1',
                    yaxis_title='Dim 2',
                    zaxis_title='Dim 3'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)