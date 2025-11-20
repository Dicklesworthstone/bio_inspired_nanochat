import streamlit as st
import os
import glob
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Check if streamlit is installed
try:
    import streamlit
except ImportError:
    print("Streamlit is not installed. Please install it with: uv add streamlit")
    exit(1)

st.set_page_config(
    page_title="Bio-Nanochat Dashboard",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for "Stripe-level" Polish
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }
    
    /* Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 { font-size: 2.5rem; margin-bottom: 1rem; background: linear-gradient(90deg, #4CAF50, #2196F3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2 { font-size: 1.8rem; margin-top: 2rem; border-bottom: 1px solid #30363D; padding-bottom: 0.5rem; }
    h3 { font-size: 1.3rem; color: #A0A0A0; }
    
    /* Cards / Containers */
    .metric-card {
        background-color: #1F242D;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58A6FF;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #8B949E;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #58A6FF;
        border-bottom: 2px solid #58A6FF;
    }
    
    /* Info Boxes */
    .stAlert {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #C9D1D9;
    }
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Bio-Inspired Nanochat")

neuroviz_dir = "runs/neuroviz"

# Wait for directory to exist
if not os.path.exists(neuroviz_dir):
    st.warning(f"Directory `{neuroviz_dir}` not found. Waiting for training to start...")
    time.sleep(2)
    st.rerun()

# -----------------------------------------------------------------------------
# Data Loading Helpers
# -----------------------------------------------------------------------------

def get_files(pattern):
    files = glob.glob(os.path.join(neuroviz_dir, pattern))
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def get_layers():
    images_dir = os.path.join(neuroviz_dir, "images")
    if not os.path.exists(images_dir):
        return []
    return [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

layers = get_layers()
if not layers:
    st.info("Waiting for first batch of visualizations (usually ~1000 steps)...")
    time.sleep(5)
    st.rerun()
    st.stop()

with st.sidebar:
    st.header("Control Panel")
    selected_layer = st.selectbox("Select Layer", layers)
    
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("Go to", [
        "Overview", 
        "Synaptic Dynamics", 
        "Anatomy of a Decision",
        "Semantic Space (3D)",
        "Functional Connectivity",
        "Interactive Petri Dish",
        "Interactive Hebbian Learning",
        "Structural Plasticity", 
        "Population Stats",
        "Genetics & Diversity",
        "Metabolism Economy"
    ])
    
    st.markdown("---")
    
    # Live Mode Toggle
    live_mode = st.toggle("Live Mode (Auto-Refresh)", value=False)
    if live_mode:
        time.sleep(2)
        st.rerun()

    if st.button("Refresh Data", type="primary"):
        st.rerun()
        
    st.markdown("""
    <div style="margin-top: 2rem; font-size: 0.8rem; color: #666;">
    v0.2.0 | Bio-Nanochat
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------

if page == "Overview":
    st.markdown("""
    <div class="metric-card">
        <h3>Welcome to the Living Brain</h3>
        <p>This dashboard visualizes the internal state of the <b>Synaptic Mixture-of-Experts</b> model. 
        Unlike standard Transformers, this model has a "metabolism" and "synaptic plasticity".</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Vitals (EKG)
    vitals_path = os.path.join(neuroviz_dir, "vitals.csv")
    if os.path.exists(vitals_path):
        try:
            df = pd.read_csv(vitals_path)
            if not df.empty:
                last_row = df.iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Global Energy", f"{last_row['energy']:.2f}", delta=None)
                col2.metric("Global Fatigue", f"{last_row['fatigue']:.2f}", delta=None, delta_color="inverse")
                col3.metric("Loss", f"{last_row['loss']:.4f}", delta=None, delta_color="inverse")
                col4.metric("Step", int(last_row['step']))
                
                # Scrolling EKG Chart
                st.markdown("### System Vitals")
                
                # Reshape for plotly
                df_melt = df.melt(id_vars=['step'], value_vars=['energy', 'fatigue', 'loss'], var_name='Metric', value_name='Value')
                
                fig = px.line(df_melt, x='step', y='Value', color='Metric', 
                              color_discrete_map={'energy': '#00CC96', 'fatigue': '#EF553B', 'loss': '#AB63FA'})
                
                fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Step",
                    yaxis_title="Value",
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading vitals: {e}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🧬 Key Biological Concepts
        
        *   **Fatigue (Presynaptic)**: Experts get "tired" if used too often. This forces the router to explore new paths (novelty seeking).
        *   **Energy (ATP)**: Experts earn energy by being useful. Low energy leads to death (Merge). High energy leads to reproduction (Split).
        *   **Plasticity (Postsynaptic)**: Weights are not static! They have a "Fast" component that learns *during inference* (Hebbian learning).
        *   **Consolidation (CaMKII)**: Important short-term memories are "written" into long-term weights if the neuron is excited enough.
        """)
        
    with col2:
        st.info(f"Monitoring directory: `{neuroviz_dir}`")
        st.write(f"Active Layers: **{len(layers)}**")
        st.write(f"Current View: **{selected_layer}**")

# -----------------------------------------------------------------------------
# Page: Synaptic Dynamics (The "Deep Dive")
# -----------------------------------------------------------------------------

elif page == "Synaptic Dynamics":
    st.header(f"Synaptic Dynamics: {selected_layer}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Weight Contribution", "Presynaptic Fatigue", "Hebbian Memory", "Expert Raster"])
    
    # --- Tab 1: Contribution ---
    with tab1:
        st.markdown("### Static vs. Bio Weights")
        st.caption("Comparing the L2 norm of the Slow Weights (Backprop) vs. the Fast Weights (Hebbian). Visible 'Fast' bars indicate active short-term memory usage.")
        
        files = get_files(f"images/{selected_layer}/{selected_layer}_contrib_*.json")
        if files:
            idx = st.slider("History (Contrib)", 0, len(files)-1, 0, format="Step -%d", key="contrib_slider")
            try:
                data = load_json(files[idx])
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data['ids'], y=data['slow_norms'], name='Slow (Static)',
                    marker_color='#4472C4'
                ))
                fig.add_trace(go.Bar(
                    x=data['ids'], y=data['fast_norms'], name='Fast (Bio)',
                    marker_color='#ED7D31'
                ))
                fig.update_layout(
                    barmode='group', 
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Weight Norm (L2)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading contribution data: {e}")
        else:
            # Fallback to PNG if JSON not found
            png_files = get_files(f"images/{selected_layer}/{selected_layer}_contrib_*.png")
            if png_files:
                idx = st.slider("History (Contrib PNG)", 0, len(png_files)-1, 0, format="Step -%d", key="contrib_png_slider")
                st.image(png_files[idx], caption=os.path.basename(png_files[idx]))
            else:
                st.warning("No contribution data found yet.")

    # --- Tab 2: Presynaptic ---
    with tab2:
        st.markdown("### Presynaptic 'Boredom' Simulation")
        st.caption("We simulate a neuron attending to the same token repeatedly (Steps 0-25) and then switching. Watch the RRP drain and the Logit Delta suppress attention.")
        
        files = get_files(f"images/{selected_layer}/{selected_layer}_presyn_*.json")
        if files:
            idx = st.slider("History (Presyn)", 0, len(files)-1, 0, format="Step -%d", key="presyn_slider")
            data = load_json(files[idx])
            
            steps = data['steps']
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                subplot_titles=("RRP (Vesicle Pool)", "Calcium (Excitement)", "Logit Adjustment"))
            
            fig.add_trace(go.Scatter(x=steps, y=data['rrp'], name="RRP", line=dict(color="#00CC96", width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=steps, y=data['calcium'], name="Calcium", line=dict(color="#FFA15A", width=3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=steps, y=data['logit_delta'], name="Delta", line=dict(color="#EF553B", width=3)), row=3, col=1)
            
            # Annotations
            fig.add_vline(x=25, line_width=1, line_dash="dash", line_color="gray")
            fig.add_annotation(x=12, y=-2, text="Attending Token A", showarrow=False, row=3, col=1, font=dict(color="gray"))
            fig.add_annotation(x=37, y=-2, text="Switched to Token B", showarrow=False, row=3, col=1, font=dict(color="gray"))

            fig.update_layout(template="plotly_dark", height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No presynaptic data found yet.")

    # --- Tab 3: Hebbian ---
    with tab3:
        st.markdown("### Hebbian Memory Trace ($H_{fast}$)")
        st.caption("A heatmap of the fast weight matrix for a single expert. Patterns here represent associations learned from the immediate context window.")
        
        files = get_files(f"images/{selected_layer}/{selected_layer}_hebbian_*.json")
        if files:
            idx = st.slider("History (Hebb)", 0, len(files)-1, 0, format="Step -%d", key="hebb_slider")
            data = load_json(files[idx])
            heatmap = np.array(data['heatmap'])
            
            fig = px.imshow(heatmap, color_continuous_scale="RdBu_r", zmin=-0.01, zmax=0.01, aspect="auto")
            fig.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Output Dim",
                yaxis_title="Input Dim"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Hebbian data found yet.")

    # --- Tab 4: Raster ---
    with tab4:
        st.markdown("### Expert Activation Raster")
        st.caption("A 'Brain Scan' showing which experts fired for the last 100 tokens. Look for sparse activation and specialization.")
        
        files = get_files(f"images/{selected_layer}/{selected_layer}_raster_*.json")
        if files:
            idx = st.slider("History (Raster)", 0, len(files)-1, 0, format="Step -%d", key="raster_slider")
            data = load_json(files[idx])
            raster = np.array(data['raster'])
            
            fig = px.imshow(raster, color_continuous_scale="Magma", aspect="auto", origin='lower')
            fig.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Token Time",
                yaxis_title="Expert ID"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No raster data found yet.")

# -----------------------------------------------------------------------------
# Page: Anatomy of a Decision
# -----------------------------------------------------------------------------

elif page == "Anatomy of a Decision":
    st.header(f"Anatomy of a Decision: {selected_layer}")
    st.markdown("""
    **Why did the Router choose Expert X?**
    
    We break down the routing logits into their biological components for a single token.
    
    *   **Content Match**: The standard dot-product similarity.
    *   **Gene Bias**: Innate preference based on genetics.
    *   **Energy Bias**: Boost for high-energy experts.
    *   **Fatigue Penalty**: Penalty for tired experts.
    """)
    
    files = get_files(f"images/{selected_layer}/{selected_layer}_decision_*.json")
    if files:
        idx = st.slider("History", 0, len(files)-1, 0, format="Step -%d", key="decision_slider")
        data = load_json(files[idx])
        
        # Data is for all experts. We want to show the Top-K winners.
        total_logits = np.array(data['total_logits'])
        top_k_idx = np.argsort(total_logits)[-3:][::-1] # Top 3
        
        # Create columns for the top 3 experts
        cols = st.columns(3)
        
        for i, expert_id in enumerate(top_k_idx):
            with cols[i]:
                st.subheader(f"Rank #{i+1}: Expert {expert_id}")
                
                # Extract components
                components = {
                    "Content": data['router_logits'][expert_id],
                    "Genetics": data['gene_bias'][expert_id],
                    "Alignment": data['align_bias'][expert_id],
                    "Energy": data['energy_bias'][expert_id],
                    "Fatigue": data['fatigue_bias'][expert_id]
                }
                
                # Waterfall chart
                fig = go.Figure(go.Waterfall(
                    name = "20", orientation = "v",
                    measure = ["relative"] * 5 + ["total"],
                    x = list(components.keys()) + ["Total"],
                    textposition = "outside",
                    text = [f"{v:.2f}" for v in components.values()] + [f"{total_logits[expert_id]:.2f}"],
                    y = list(components.values()) + [total_logits[expert_id]],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                ))
                
                fig.update_layout(
                    title = "Logit Composition",
                    showlegend = False,
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.warning("No decision data found yet.")

# -----------------------------------------------------------------------------
# Page: Semantic Space (3D)
# -----------------------------------------------------------------------------

elif page == "Semantic Space (3D)":
    st.header(f"Semantic Space (3D): {selected_layer}")
    st.markdown("""
    **Where does the Token live?**
    
    We project the **Token's Router Probe** and all **Expert Embeddings** into 3D space.
    
    *   **🔴 Red Star**: The current Token.
    *   **⚪ Grey Dots**: The Experts.
    *   **🔵 Blue Halo**: The Experts that were actually chosen (Gated).
    
    This shows if the router is picking experts that are *semantically close* to the token (good alignment) or if other biases (Fatigue/Energy) are forcing it to pick distant experts.
    """)
    
    files = get_files(f"images/{selected_layer}/{selected_layer}_semantic_*.json")
    if files:
        idx = st.slider("History", 0, len(files)-1, 0, format="Step -%d", key="semantic_slider")
        data = load_json(files[idx])
        
        token_x = data['token_x']
        token_y = data['token_y']
        token_z = data.get('token_z', 0.0) # Fallback for old 2D data
        
        exp_x = data['experts_x']
        exp_y = data['experts_y']
        exp_z = data.get('experts_z', [0.0]*len(exp_x))
        
        gates = np.array(data['gates'])
        
        fig = go.Figure()
        
        # All experts
        fig.add_trace(go.Scatter3d(
            x=exp_x, y=exp_y, z=exp_z,
            mode='markers',
            marker=dict(
                size=5,
                color=gates,
                colorscale=[[0, 'grey'], [0.01, 'grey'], [0.01, '#2196F3'], [1, '#2196F3']],
                showscale=False,
                opacity=0.6
            ),
            text=[f"Expert {i}<br>Gate: {g:.2f}" for i, g in enumerate(gates)],
            hoverinfo='text',
            name='Experts'
        ))
        
        # Selected experts (Halo)
        selected_mask = gates > 0
        if np.any(selected_mask):
            fig.add_trace(go.Scatter3d(
                x=np.array(exp_x)[selected_mask],
                y=np.array(exp_y)[selected_mask],
                z=np.array(exp_z)[selected_mask],
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgba(0,0,0,0)',
                    line=dict(color='#2196F3', width=5)
                ),
                hoverinfo='skip',
                name='Selected'
            ))
        
        # Token
        fig.add_trace(go.Scatter3d(
            x=[token_x], y=[token_y], z=[token_z],
            mode='markers',
            marker=dict(size=12, symbol='diamond', color='#FF5252'),
            name='Current Token'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            ),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No semantic space data found yet.")

# -----------------------------------------------------------------------------
# Page: Functional Connectivity
# -----------------------------------------------------------------------------

elif page == "Functional Connectivity":
    st.header(f"Functional Connectivity: {selected_layer}")
    st.markdown("""
    **The Connectome.**
    
    This graph shows which experts fire together.
    *   **Nodes**: Experts.
    *   **Edges**: Co-activation frequency.
    
    Clusters represent **Functional Modules** (e.g., a group of experts that specialize in Python code or French grammar).
    """)
    
    files = get_files(f"images/{selected_layer}/{selected_layer}_connectivity_*.json")
    if files:
        idx = st.slider("History", 0, len(files)-1, 0, format="Step -%d", key="connect_slider")
        data = load_json(files[idx])
        adj = np.array(data['adjacency'])
        
        # Build graph for Plotly
        # We use a circular layout for simplicity and stability
        E = adj.shape[0]
        
        # Filter weak edges
        threshold = 0.1
        rows, cols = np.where(adj > threshold)
        
        edge_x = []
        edge_y = []
        
        # Circular layout
        radius = 10
        node_x = [radius * np.cos(2*np.pi*i/E) for i in range(E)]
        node_y = [radius * np.sin(2*np.pi*i/E) for i in range(E)]
        
        for r, c in zip(rows, cols):
            if r >= c:
                continue # Undirected
            x0, y0 = node_x[r], node_y[r]
            x1, y1 = node_x[c], node_y[c]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='#2196F3',
                size=10,
                line_width=2),
            text=[str(i) for i in range(E)],
            textposition="top center"
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Expert Co-Activation Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No connectivity data found yet.")

# -----------------------------------------------------------------------------
# Page: Interactive Petri Dish
# -----------------------------------------------------------------------------

elif page == "Interactive Petri Dish":
    st.header("🧫 Interactive Petri Dish")
    st.markdown("""
    **Simulate Presynaptic Dynamics in Real-Time.**
    
    Adjust the biological parameters below to see how a synapse responds to "boredom" (repeated stimulation).
    *   **Tau RRP**: How fast the vesicle pool refills.
    *   **Alpha Ca**: How much calcium enters per spike.
    *   **Tau Ca**: How fast calcium decays.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Bio-Parameters")
        tau_rrp = st.slider("Tau RRP (Refill Time)", 1.0, 100.0, 40.0)
        alpha_ca = st.slider("Alpha Ca (Influx)", 0.01, 1.0, 0.25)
        tau_c = st.slider("Tau Ca (Decay)", 1.0, 20.0, 4.0)
        complexin = st.slider("Complexin Bias (Clamp)", 0.0, 1.0, 0.5)
        
    with col2:
        # Run simulation (Pure Python implementation of SynapticPresyn logic)
        T = 100
        # Stimulus: 50 steps ON, 50 steps OFF
        stimulus = np.zeros(T)
        stimulus[:50] = 20.0 # High logit
        
        # State
        C = 0.0
        RRP = 1.0
        
        # History
        rrp_hist = []
        c_hist = []
        release_hist = []
        
        import math
        
        for t in range(T):
            # 1. Calcium Influx
            influx = np.log(1 + np.exp(stimulus[t])) # Softplus
            rho_c = math.exp(-1.0 / tau_c)
            C = rho_c * C + alpha_ca * influx
            
            # 2. Release Prob
            # Simplified Syt model
            syt = C / (C + 0.4)
            p_release = 1.0 / (1.0 + np.exp(-(3.0 * syt - 2.0 * complexin))) # Sigmoid
            
            # 3. Release
            release = p_release * RRP
            
            # 4. Refill
            rho_r = math.exp(-1.0 / tau_rrp)
            RRP = rho_r * RRP - release + 0.04 * (1.0 - RRP) # Simplified refill
            
            rrp_hist.append(RRP)
            c_hist.append(C)
            release_hist.append(release)
            
        # Plot
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(y=stimulus, name="Stimulus", fill='tozeroy'), row=1, col=1)
        fig.add_trace(go.Scatter(y=c_hist, name="Calcium", line=dict(color="orange")), row=2, col=1)
        fig.add_trace(go.Scatter(y=rrp_hist, name="RRP (Pool)", line=dict(color="green")), row=3, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Page: Interactive Hebbian Learning
# -----------------------------------------------------------------------------

elif page == "Interactive Hebbian Learning":
    st.header("🧠 Interactive Hebbian Learning")
    st.markdown("""
    **Simulate Short-Term Memory (Fast Weights).**
    
    The model uses a "Fast Weight" matrix $H_{fast}$ to store associations from the recent context.
    
    *   **Rule**: $H_{fast} \leftarrow \\rho H_{fast} + \eta (U \cdot V^T)$
    *   **Intuition**: "Neurons that fire together, wire together."
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        eta = st.slider("Learning Rate (Eta)", 0.01, 0.5, 0.1)
        rho = st.slider("Decay Rate (Rho)", 0.5, 1.0, 0.9)
        steps = st.slider("Sequence Length", 5, 50, 20)
        
        st.subheader("Input Sequence")
        pattern_type = st.radio("Pattern", ["Repeated (A A A...)", "Alternating (A B A B...)", "Random"])
        
    with col2:
        # Simulation
        dim = 10
        # Define patterns
        np.random.seed(42)
        pat_A = np.random.randn(dim)
        pat_A /= np.linalg.norm(pat_A)
        pat_B = np.random.randn(dim)
        pat_B /= np.linalg.norm(pat_B)
        
        # Generate sequence
        seq = []
        labels = []
        for t in range(steps):
            if pattern_type == "Repeated (A A A...)":
                seq.append(pat_A)
                labels.append("A")
            elif pattern_type == "Alternating (A B A B...)":
                if t % 2 == 0:
                    seq.append(pat_A)
                    labels.append("A")
                else:
                    seq.append(pat_B)
                    labels.append("B")
            else:
                if np.random.rand() > 0.5:
                    seq.append(pat_A)
                    labels.append("A")
                else:
                    seq.append(pat_B)
                    labels.append("B")
                    
        # Run Hebbian Dynamics
        # Simplified: y = x (W + H). We assume W=Identity for clarity.
        # H updates based on x * y^T (Oja-like or simple Hebb)
        # In code: U update, V update, H update.
        # Simplified for demo: H += eta * x * y^T
        
        H = np.zeros((dim, dim))
        W = np.eye(dim)
        
        magnitudes = []
        alignments_A = []
        alignments_B = []
        
        for x in seq:
            # Forward
            y = x @ (W + H)
            
            # Measure
            mag = np.linalg.norm(y)
            magnitudes.append(mag)
            
            # Alignment with patterns (Recall)
            align_A = np.dot(y / (mag + 1e-8), pat_A)
            align_B = np.dot(y / (mag + 1e-8), pat_B)
            alignments_A.append(align_A)
            alignments_B.append(align_B)
            
            # Update
            # Simple Hebb: dH = eta * outer(x, y)
            # Decay
            H = rho * H + eta * np.outer(x, y)
            
        # Plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Output Magnitude (Familiarity)", "Pattern Alignment (Recall)"))
        
        x_axis = list(range(steps))
        
        # Magnitude
        fig.add_trace(go.Scatter(x=x_axis, y=magnitudes, name="Output Mag", line=dict(color="#F4D03F", width=3)), row=1, col=1)
        
        # Alignment
        fig.add_trace(go.Scatter(x=x_axis, y=alignments_A, name="Match A", line=dict(color="#2ECC71", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=alignments_B, name="Match B", line=dict(color="#3498DB", width=2)), row=2, col=1)
        
        # Add labels to x-axis
        fig.update_xaxes(ticktext=labels, tickvals=x_axis, row=2, col=1)
        
        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Final H Matrix Norm: {np.linalg.norm(H):.2f}")

# -----------------------------------------------------------------------------
# Page: Structural Plasticity
# -----------------------------------------------------------------------------

elif page == "Structural Plasticity":
    st.header(f"Lineage Tree: {selected_layer}")
    st.markdown("""
    **Structural Plasticity in Action.**
    
    *   **🟢 Split (Birth)**: A healthy, high-energy expert clones itself to handle more load.
    *   **🟣 Merge (Death)**: A starving, low-utility expert is absorbed by a neighbor.
    """)
    
    # Look for HTML files first
    html_files = get_files(f"lineage/{selected_layer}_lineage_*.html")
    if html_files:
        latest_html = html_files[0]
        st.caption(f"Latest interactive lineage: {os.path.basename(latest_html)}")
        with open(latest_html, 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.info("No interactive lineage HTML found yet.")

# -----------------------------------------------------------------------------
# Page: Population Stats
# -----------------------------------------------------------------------------

elif page == "Population Stats":
    st.header(f"Population Stats: {selected_layer}")
    
    tab1, tab2, tab3 = st.tabs(["Expert Map", "Distributions", "Radar Charts"])
    
    with tab1:
        st.markdown("**The 'Brain Map'**")
        st.caption("Each dot is an Expert. Position = Semantic Specialization. Size = Utilization. Color = Energy.")
        map_files = get_files(f"images/{selected_layer}/{selected_layer}_map_*.png")
        if map_files:
            idx = st.slider("History", 0, len(map_files)-1, 0, format="Step -%d", key="map_slider")
            st.image(map_files[idx], caption=os.path.basename(map_files[idx]))
        else:
            st.info("No expert maps found yet.")
            
    with tab2:
        st.markdown("**Population Distributions**")
        hist_files = get_files(f"images/{selected_layer}/{selected_layer}_hists_*.png")
        if hist_files:
            idx = st.slider("History", 0, len(hist_files)-1, 0, format="Step -%d", key="hist_slider")
            st.image(hist_files[idx], caption=os.path.basename(hist_files[idx]))
        else:
            st.info("No histograms found yet.")
            
    with tab3:
        st.markdown("**Top Experts Radar**")
        radar_files = get_files(f"images/{selected_layer}/{selected_layer}_radar_*.png")
        if radar_files:
            idx = st.slider("History", 0, len(radar_files)-1, 0, format="Step -%d", key="radar_slider")
            st.image(radar_files[idx], caption=os.path.basename(radar_files[idx]))
        else:
            st.info("No radar charts found yet.")

# -----------------------------------------------------------------------------
# Page: Genetics & Diversity
# -----------------------------------------------------------------------------

elif page == "Genetics & Diversity":
    st.header(f"Genetics & Diversity: {selected_layer}")
    st.markdown("""
    **Evolutionary Drift**
    
    Each expert has a unique "genome" ($Xi$) that determines its biological properties.
    We visualize how the population diversifies into different phenotypes.
    
    *   **X-axis**: Fatigue Rate (How fast they tire)
    *   **Y-axis**: Energy Refill (How fast they recover)
    *   **Color**: CaMKII Gain (Learning Rate)
    *   **Size**: Utilization (Success)
    """)
    
    files = get_files(f"images/{selected_layer}/{selected_layer}_genetics_*.json")
    if files:
        idx = st.slider("History", 0, len(files)-1, 0, format="Step -%d", key="gene_slider")
        data = load_json(files[idx])
        
        df = pd.DataFrame({
            "Fatigue Rate": data['fatigue_rate'],
            "Energy Refill": data['energy_refill'],
            "CaMKII Gain": data['camkii_gain'],
            "Utilization": data['utilization']
        })
        
        fig = px.scatter(
            df, 
            x="Fatigue Rate", 
            y="Energy Refill", 
            color="CaMKII Gain", 
            size="Utilization",
            hover_data=["CaMKII Gain"],
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No genetics data found yet.")

# -----------------------------------------------------------------------------
# Page: Metabolism Economy
# -----------------------------------------------------------------------------

elif page == "Metabolism Economy":
    st.header(f"Metabolism Economy: {selected_layer}")
    st.markdown("""
    **The Energy Market**
    
    Experts earn Energy (ATP) by being useful and spend it by firing.
    This chart shows the distribution of wealth (Energy) across the population.
    
    *   **Inequality**: A steep curve means a few "rich" experts dominate.
    *   **Poverty Line**: Experts near 0 energy are at risk of death (Merge).
    """)
    
    files = get_files(f"images/{selected_layer}/{selected_layer}_metabolism_*.json")
    if files:
        idx = st.slider("History", 0, len(files)-1, 0, format="Step -%d", key="meta_slider")
        data = load_json(files[idx])
        
        energy = np.array(data['energy'])
        ids = np.array(data['ids'])
        
        # Create a bar chart sorted by energy
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(energy))),
            y=energy,
            marker=dict(color=energy, colorscale='Plasma'),
            text=ids,
            hovertemplate="Expert ID: %{text}<br>Energy: %{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Wealth Distribution (Energy)",
            xaxis_title="Experts (Sorted by Wealth)",
            yaxis_title="Energy Level",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add poverty line
        fig.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Starvation Risk")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metabolism data found yet.")
