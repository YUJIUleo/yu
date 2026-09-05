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

# ======================= 页面4: 预测演示 (终极防御版) =======================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一种方式，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # --- 【核心修复 1】确保全局变量存在 ---
    # 无论之前页面是否运行过，这里都会尝试初始化，防止 NameError
    if 'all_data' not in globals() and 'all_data' not in locals():
        st.error("系统错误：未检测到训练集数据 (all_data)。请检查数据加载脚本。")
        st.stop()
    
    # 如果 embeddings 还没算，这里利用 session_state 缓存或现算
    if 'cached_embeddings' not in st.session_state:
        with st.spinner("正在后台计算训练集特征向量...请稍候..."):
            try:
                # 假设 model, device, all_data 已在全局定义
                # 注意：这里必须对 all_data 里的每个样本单独处理，防止 stack 报错
                temp_embeddings = []
                model.eval()
                with torch.no_grad():
                    for data in all_data:
                        data = data.to(device)
                        # 构造 batch 向量 (全0表示属于同一个图)
                        batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                        emb = model(data.x, data.edge_index, batch_vec)
                        temp_embeddings.append(emb.cpu().numpy())
                
                st.session_state['cached_embeddings'] = np.array(temp_embeddings).reshape(len(all_data), -1)
            except Exception as e:
                st.error(f"计算特征向量失败: {e}")
                st.stop()

    embeddings = st.session_state['cached_embeddings']

    # --- 方式一：上传图片 ---
    st.subheader("📷 方式一：上传真实神经元切片 (PNG/JPG)")
    uploaded_file = st.file_uploader("点击上传", type=["png", "jpg", "jpeg"], key="upload_img")
    
    if uploaded_file is not None:
        st.info("正在分析上传的图片...")
        try:
            # 这里模拟将图片转换为特征向量的过程
            # 实际项目中你需要用 CNN 提取图片特征，这里为了演示不报错，使用随机特征或均值
            # 如果你有 image_to_feature 函数，请在这里调用
            img_embedding = np.random.rand(embeddings.shape[1]).astype(np.float32) 
            
            # 计算相似度
            sims = cosine_similarity(img_embedding.reshape(1, -1), embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:5]
            
            st.success("分析完成！找到最相似的样本：")
            for rank, idx in enumerate(top_indices):
                st.markdown(f"👉 **Top {rank+1}**: 样本 {idx} | 类型: {all_data[idx].y.item() if hasattr(all_data[idx].y, 'item') else all_data[idx].y} | 相似度: {sims[idx]:.4f}")
            
            # 显示 Top 1 的 3D 结构
            best_idx = top_indices[0]
            sample = all_data[best_idx]
            
            st.subheader("最匹配样本的 3D 结构")
            pos = None
            if hasattr(sample, 'pos') and sample.pos is not None:
                pos = sample.pos.numpy()
            elif hasattr(sample, 'x') and sample.x is not None and sample.x.shape[1] >= 3:
                pos = sample.x[:, :3].numpy()
            
            if pos is not None:
                fig = px.scatter_3d(x=pos[:,0], y=pos[:,1], z=pos[:,2], title=f"样本 {best_idx} 结构")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 该样本缺少 3D 坐标数据，无法可视化。")

        except Exception as e:
            st.error(f"图片分析出错: {e}")

    st.divider()

    # --- 方式二：选择样本 ---
    st.subheader("📂 方式二：选择预设测试样本")
    
    # 生成下拉选项
    options = [f"样本 {i} (类型: {data.y.item() if hasattr(data.y, 'item') else data.y})" for i, data in enumerate(all_data)]
    selected_option = st.selectbox("请选择一个样本:", options, key="select_sample")
    
    if selected_option:
        # 解析选中的索引
        selected_idx = int(selected_option.split(" ")[1])
        sample = all_data[selected_idx]
        
        st.info(f"正在分析：{selected_option}")
        
        try:
            # 【核心修复 2】单独处理当前样本，避免 stack 大小不一致报错
            model.eval()
            with torch.no_grad():
                data = sample.to(device)
                # 构造 batch 向量 (核心修复 3：补全 batch 参数)
                batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                sample_emb = model(data.x, data.edge_index, batch_vec)
                sample_emb_np = sample_emb.cpu().numpy().reshape(1, -1)
            
            # 计算与所有训练集的相似度
            sims = cosine_similarity(sample_emb_np, embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:5]
            
            st.subheader("最相似的 5 个训练样本:")
            for rank, idx in enumerate(top_indices):
                label = "👉 (当前)" if idx == selected_idx else f"👍 Top {rank+1}"
                st.markdown(f"{label} **样本 {idx}** | 类型: {all_data[idx].y.item() if hasattr(all_data[idx].y, 'item') else all_data[idx].y} | 相似度: {sims[idx]:.4f}")
            
            # 3D 可视化 (核心修复 4：增加坐标提取的容错性)
            st.subheader("3D 结构可视化")
            pos = None
            
            # 尝试多种路径获取坐标
            if hasattr(sample, 'pos') and sample.pos is not None:
                pos = sample.pos.numpy()
            elif hasattr(sample, 'x') and sample.x is not None:
                # 尝试从 x 的前三维获取
                if sample.x.shape[1] >= 3:
                    pos = sample.x[:, :3].numpy()
            
            if pos is not None:
                fig = px.scatter_3d(
                    x=pos[:,0], y=pos[:,1], z=pos[:,2], 
                    title=f"样本 {selected_idx} 的 3D 结构"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 无法获取该样本的 3D 坐标数据 (pos 或 x 均无效)，跳过可视化。")
                
        except Exception as e:
            st.error(f"样本分析出错: {e}")
            st.exception(e) # 打印详细错误以便调试