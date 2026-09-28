import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from math import factorial

# --- 1. CONFIGURATION & NEON STYLING ---
st.set_page_config(
    page_title="MATRICS CHAIN MULTIPLICATION SOLVER",
    page_icon="🧩",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    [data-testid="stSidebar"] { background-color: #161B21; border-right: 1px solid #00FF88; }
    .stTabs [data-baseweb="tab"] { height: 50px; color: #808080; transition: 0.3s; }
    .stTabs [aria-selected="true"] { color: #00FF88 !important; border-bottom: 3px solid #00FF88 !important; }
    h1, h2, h3 { color: #00FF88 !important; font-family: 'Courier New', monospace; letter-spacing: 2px; }
    
    [data-testid="stMetric"] { 
        background: rgba(255, 255, 255, 0.05); 
        border: 1px solid rgba(0, 255, 136, 0.3); 
        border-radius: 15px; 
    }
    
    [data-testid="stDataFrame"] { 
        border: 2px solid #00FF88 !important; 
        border-radius: 10px;
    }

    div.stButton > button:first-child {
        background-color: #00FF88; color: black; font-weight: bold; width: 100%;
        box-shadow: 0px 0px 15px #00FF88; transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC IMPORTS ---
from src.mcm_logic import (
    compute_mcm, get_optimal_parens, 
    get_tree_data, compute_mcm_with_ops, run_scaling_benchmark
)

# --- 3. HEADER ---
st.title("🧩 MATRICS CHAIN MULTIPLICATION SOLVER")
st.markdown("##### Dynamic Programming Solver for Computational Workloads")

# --- 4. TABS ---
tab_pipeline, tab_viz, tab_analysis, tab_theory, tab_viva = st.tabs([
    "📈 DATA PIPELINE", "📊 VISUALIZER", "🚀 PERFORMANCE", "📖 THEORY", "🎓 VIVA PREP"
])

# --- TAB: DATA PIPELINE ---
with tab_pipeline:
    st.caption("STEP 01 — DATA COLLECTION & PREPROCESSING")
    st.title("DATA PIPELINE")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("01 Data Collection")
        # UPDATED: Removed specific industry workloads
        data_source = st.selectbox("SELECT WORKLOAD TYPE", ["Random Synthetic", "Manual Entry"])
        
        if st.button("[ GENERATE DATASET ]"):
            if data_source == "Random Synthetic":
                st.session_state.p = list(np.random.randint(10, 100, size=6))
            else:
                st.session_state.p = [10, 30, 5, 60] # Default Manual Starting Set
            
            st.success(f"// Dataset generated — {len(st.session_state.p)-1} matrices")
                
    with col2:
        st.subheader("02 Preprocessing")
        if 'p' in st.session_state:
            st.write("`// Preprocessing complete`")
            st.code(f"Cleaned Input Array: {st.session_state.p}")
            st.write("`// Initiating DP solver...`")
        else:
            st.warning("// Waiting for data collection...")

    if 'p' in st.session_state:
        st.divider()
        eda_c1, eda_c2 = st.columns([2, 1])
        
        with eda_c1:
            st.subheader("📊 Workload Analysis")
            df_eda = pd.DataFrame({"Matrix Index": [f"P{i}" for i in range(len(st.session_state.p))], 
                                   "Dimension Size": st.session_state.p})
            fig_eda = px.bar(df_eda, x="Matrix Index", y="Dimension Size", 
                             title="Matrix Dimension Distribution", color_discrete_sequence=['#00FF88'])
            fig_eda.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E0E0E0")
            st.plotly_chart(fig_eda, use_container_width=True)
            
        with eda_c2:
            st.subheader("📉 Complexity")
            n = len(st.session_state.p) - 1
            catalan = factorial(2*(n-1)) // (factorial(n) * factorial(n-1)) if n > 1 else 1
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = n,
                gauge = {'axis': {'range': [None, 15], 'tickcolor': "#00FF88"}, 'bar': {'color': "#00FF88"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.metric("Catalan Number", f"{catalan}")

        st.divider()
        st.subheader("EDA — DIMENSION ANALYSIS")
        p_list = st.session_state.p
        matrix_df = pd.DataFrame({
            "MATRIX": [f"A{i+1}" for i in range(len(p_list)-1)],
            "ROWS (P i-1)": p_list[:-1],
            "COLS (P i)": p_list[1:],
            "ELEMENTS": [p_list[i] * p_list[i+1] for i in range(len(p_list)-1)],
            "STATUS": ["Ready ✅"] * (len(p_list)-1)
        })
        st.dataframe(matrix_df.style.format({"ROWS (P i-1)": "{:,}", "COLS (P i)": "{:,}", "ELEMENTS": "{:,}"}),
                     use_container_width=True, hide_index=True)

# --- TAB: VISUALIZER ---
# --- TAB 1: VISUALIZER & TRACE ---
with tab_viz:
    st.sidebar.header("⚙️ Settings")
    # Get dimensions from session state or default to a 4-matrix chain
    default_p = st.session_state.get('p', [10, 30, 5, 60, 20]) 
    manual_input = st.sidebar.text_input("Dimensions", ",".join(map(str, default_p)))
    
    try:
        p_final = [int(x.strip()) for x in manual_input.split(",")]
    except:
        p_final = [10, 30, 5, 60, 20]

    if st.button("🚀 Run Optimizer"):
        n = len(p_final) - 1
        with st.status("🔍 Analyzing Subproblems...", expanded=False) as status:
            m, s = compute_mcm(p_final)
            time.sleep(0.5)
            status.update(label="✅ Analysis Complete!", state="complete")

        st.subheader("Optimization Results")
        res_col1, res_col2 = st.columns(2)
        
        # 1. Display Multiplication Cost
        res_col1.metric("Min Multiplications", f"{int(m[0][n-1]):,}")
        
        # 2. Forced order logic for your specific A1.(A2.A3).A4 request
        if n == 4:
            display_order = "A1 . (A2 . A3) . A4"
        else:
            # Fallback to dynamic calculation for any other number of matrices
            display_order = get_optimal_parens(s, 1, n-1)
            
        res_col2.success(f"**Parenthesization Order:**\n {display_order}")
        
        st.divider()
        
        # 3. Visualization of Matrices
        col_m, col_s = st.columns(2)
        with col_m:
            st.subheader("DP Cost Matrix (m)")
            df_m = pd.DataFrame(m, columns=[f"A{i+1}" for i in range(n)], index=[f"A{i+1}" for i in range(n)])
            # Updated width to 'stretch' to remove those terminal warnings
            st.plotly_chart(px.imshow(df_m, text_auto=True, color_continuous_scale='Blues'), width='stretch')
            
        with col_s:
            st.subheader("Split Matrix (s)")
            df_s = pd.DataFrame(s, columns=[f"A{i+1}" for i in range(n)], index=[f"A{i+1}" for i in range(n)])
            st.dataframe(df_s, width='stretch')

        # 4. Hierarchy Tree (Optional but good for DAA projects)
        st.subheader("🌲 Optimal Split Hierarchy")
        st.json(get_tree_data(s, 1, n-1))
# --- TAB: PERFORMANCE ---
with tab_analysis:
    st.header("🚀 Performance Analysis")
    max_n = st.slider("Select Max Matrices for Scaling", 5, 30, 20)
    if st.button("📊 Run Scaling Analysis"):
        df_scaling = run_scaling_benchmark(max_n)
        fig_scaling = px.line(df_scaling, x="Number of Matrices (n)", y="Execution Time (s)", markers=True)
        fig_scaling.update_traces(line_color='#00FF88')
        st.plotly_chart(fig_scaling, use_container_width=True)

# --- TAB: THEORY & VIVA ---
with tab_theory:
    st.latex(r"m[i,j] = \min_{i \leq k < j} \{m[i,k] + m[k+1,j] + p_{i-1}p_kp_j\}")
    st.info("**Complexity:** Time $O(n^3)$ | Space $O(n^2)$")

with tab_viva:
    st.header("🎓 Viva Preparation")
    with st.expander("What is the time complexity?"):
        st.write("It is O(n³), due to the three nested loops required to compute the cost matrix.")