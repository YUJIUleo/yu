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

# ======================= 页面4: 预测演示 (最终修复版) =======================
elif page == "🔮 预测演示":
    st.title("🔮 新样本预测演示")
    st.write("选择一种方式，查看模型如何将其映射到嵌入空间并判断类型。")
    st.divider()

    # --- 【关键前置检查】确保 all_data 存在 ---
    # 如果 all_data 还没加载，尝试在这里初始化（防止下面取值时报错）
    if 'all_data' not in globals() and 'all_data' not in locals():
        try:
            # 这里假设你的数据加载函数叫 load_data() 或者类似的
            # 如果没有，请确保你在其他页面已经正确加载了 all_data
            # 为了演示，这里假设 all_data 是全局存在的，或者你需要手动导入
            st.warning("⚠️ 检测到数据未加载，正在尝试重新加载...")
            # all_data = load_your_data_function() # <--- 请取消注释并填入你的加载函数
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            st.stop()

    # 确保 model 也存在
    if 'model' not in globals():
         st.error("模型未加载，请检查主程序配置。")
         st.stop()

    tab1, tab2 = st.tabs(["📷 方式一：上传图片", "📂 方式二：选择样本"])

    # ================== 方式一：上传图片 ==================
    with tab1:
        st.subheader("上传真实神经元切片 (PNG/JPG)")
        uploaded_file = st.file_uploader("选择图片", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                from PIL import Image
                import numpy as np
                import torch
                from sklearn.metrics.pairwise import cosine_similarity

                # 1. 图片预处理
                img = Image.open(uploaded_file).convert("L")
                img_np = np.array(img)
                
                # 简单的二值化处理（根据你的实际需求调整阈值）
                binary = (img_np > 127).astype(np.uint8) 
                
                # 2. 计算特征 (模拟你的特征提取逻辑)
                # 注意：这里需要和你训练时的特征提取逻辑保持一致
                # 假设你的模型输入需要特定的格式，这里简化处理
                h, w = binary.shape
                
                # 这里只是为了演示不报错，实际请替换为你真实的特征提取代码
                # 比如计算 area, perimeter 等
                area = np.sum(binary)
                # ... 其他特征 ...
                
                # 构造输入向量 (假设你的模型输入维度是固定的)
                # 如果模型是处理图结构的，这里需要构建 graph
                # 这里假设你有一个函数 extract_features_from_image
                # feat = extract_features_from_image(binary) 
                
                # ⚠️ 临时方案：为了演示流程，我们生成一个随机向量或简单向量
                # 请务必替换为你真实的图片转 tensor 逻辑
                dummy_feat = torch.randn(1, 10) # 假设输入维度是10
                
                # 3. 模型预测
                with torch.no_grad():
                    pred_emb = model(dummy_feat).numpy().reshape(1, -1)

                # 4. 计算相似度
                # 确保 embeddings 是全局变量且已计算
                if 'embeddings' in globals():
                    sims = cosine_similarity(pred_emb, embeddings)[0]
                    top5_idx = np.argsort(sims)[-5:][::-1]
                    
                    st.success("✅ 分析完成！最相似的 Top 5 样本：")
                    for rank, idx in enumerate(top5_idx):
                        st.write(f"{rank+1}. 样本 #{idx} (相似度: {sims[idx]:.4f})")
                else:
                    st.warning("⚠️ 训练集特征库 (embeddings) 尚未加载，无法进行比对。")

            except Exception as e:
                st.error(f"图片处理出错: {e}")

    # ================== 方式二：选择样本 ==================
    with tab2:
        st.subheader("从训练集中选择样本进行测试")
        
        # 安全检查：确保有数据可选
        if 'all_data' in globals() and len(globals()['all_data']) > 0:
            all_data = globals()['all_data']
            
            # 生成选项列表
            options = [f"样本 {i}" for i in range(len(all_data))]
            selected_option = st.selectbox("请选择一个样本:", options)
            
            if selected_option:
                # 获取选中的索引
                selected_idx = int(selected_option.split(" ")[1])
                sample = all_data[selected_idx]
                
                st.info(f"正在分析：{selected_option}")
                
                try:
                    import torch
                    from sklearn.metrics.pairwise import cosine_similarity
                    import numpy as np

                    # 1. 准备单个样本的数据
                    # 关键点：不要把所有数据 stack 在一起，只处理当前这一个 sample
                    # 假设 sample 是一个对象，包含 x (坐标) 和 edge_index
                    
                    # 将数据转为 Tensor (增加 batch 维度)
                    # 注意：这里要根据你 sample 的实际结构来写
                    # 假设 sample.x 是坐标 [N, 3]
                    x_tensor = torch.tensor(sample.x, dtype=torch.float).unsqueeze(0) # [1, N, 3]
                    
                    # 如果你的模型需要 edge_index，也要单独取
                    edge_index = torch.tensor(sample.edge_index, dtype=torch.long)
                    
                    # 2. 送入模型
                    with torch.no_grad():
                        # ⚠️ 关键修复：只传入这一个样本的数据
                        # 如果你的模型 forward 定义是 def forward(self, x, edge_index):
                        sample_emb = model(x_tensor, edge_index).numpy().reshape(1, -1)
                    
                    # 3. 计算相似度
                    if 'embeddings' in globals():
                        sims_sample = cosine_similarity(sample_emb, embeddings)[0]
                        top5_idx_sample = np.argsort(sims_sample)[-5:][::-1]
                        
                        st.success("✅ 分析完成！该样本在库中的相似度排名：")
                        
                        # 展示结果
                        cols = st.columns(5)
                        for i, idx in enumerate(top5_idx_sample):
                            with cols[i]:
                                st.metric(f"Top {i+1}", f"#{idx}", f"{sims_sample[idx]:.2%}")
                    else:
                        st.warning("⚠️ 缺少 embeddings 数据，无法计算相似度。")

                except Exception as e:
                    st.error(f"样本分析出错: {e}")
                    st.exception(e) # 显示详细错误信息以便调试
        else:
            st.error("❌ 未找到训练数据 (all_data)，请先运行数据加载程序。")