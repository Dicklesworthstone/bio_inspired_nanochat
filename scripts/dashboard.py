import streamlit as st
import os
import glob
import time
import json
import numpy as np
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
        "Structural Plasticity", 
        "Population Stats"
    ])
    
    st.markdown("---")
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
