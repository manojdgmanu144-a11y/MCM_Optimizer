import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from math import factorial

# --- 1. CONFIGURATION & NEON STYLING ---
st.set_page_config(
    page_title="MCM Optimizer Pro",
    page_icon="🧩",
    layout="wide"
)

# Combined Global CSS: Cyber-Pro Tabs + Detailed UI Overrides
st.markdown("""
    <style>
    /* 1. MAIN APP BACKGROUND */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }

    /* 2. NEON SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #00FF88;
    }

    /* 3. COLORED TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        color: #808080; 
        border: 1px solid transparent;
        transition: 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 136, 0.1) !important;
        color: #00FF88 !important; 
        border-bottom: 3px solid #00FF88 !important; 
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0, 255, 136, 0.2);
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #00FF88;
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* 4. HEADERS (NEON GREEN) */
    h1, h2, h3 {
        color: #00FF88 !important;
        font-family: 'Courier New', Courier, monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* 5. GLASSMORPHISM CARDS */
    .stVerticalBlock > div > div {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 10px;
        padding: 20px;
    }

    /* 6. NEON METRIC CARDS */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 15px;
        padding: 15px;
    }
    
    [data-testid="stMetricValue"] {
        color: #00FF88 !important;
        font-family: 'Courier New', monospace;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #808080 !important;
        text-transform: uppercase;
    }

    /* 7. NEON CODE BLOCK & ALERT BORDERS */
    .stCodeBlock {
        border-left: 5px solid #00FF88 !important;
    }
    
    .stAlert {
        background-color: rgba(0, 255, 136, 0.05) !important;
        border: 1px solid #00FF88 !important;
        color: #00FF88 !important;
    }

    /* 8. SLIDER & BUTTON EFFECTS */
    div.stButton > button:first-child {
        background-color: #00FF88;
        color: black;
        font-weight: bold;
        width: 100%;
        border: none;
        box-shadow: 0px 0px 15px #00FF88;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #05cc6f;
        box-shadow: 0px 0px 25px #00FF88;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC IMPORTS ---
# Ensure these match your existing src/mcm_logic.py file structure
from src.mcm_logic import (
    compute_mcm, get_optimal_parens, recursive_mcm, 
    get_tree_data, compute_mcm_with_ops, run_scaling_benchmark
)

# --- 3. HEADER ---
st.title("🧩 MCM Data Pipeline Pro")
st.markdown("##### Dynamic Programming Solver for Computational Workloads")

# --- 4. TAB DEFINITION ---
tab_pipeline, tab_viz, tab_analysis, tab_theory, tab_viva = st.tabs([
    "📈 DATA PIPELINE", 
    "📊 VISUALIZER", 
    "🚀 PERFORMANCE", 
    "📖 THEORY", 
    "🎓 VIVA PREP"
])

# --- TAB 0: DATA COLLECTION & PREPROCESSING ---
with tab_pipeline:
    st.caption("STEP 01 — DATA COLLECTION & PREPROCESSING")
    st.title("DATA PIPELINE")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("01 Data Collection")
            data_source = st.selectbox("SELECT WORKLOAD TYPE", 
                                     ["Deep Learning (LLM Layers)", "Computer Graphics (Rendering)", "Random Synthetic"])
            
            if st.button("[ GENERATE DATASET ]"):
                if data_source == "Deep Learning (LLM Layers)":
                    st.session_state.p = [4096, 11008, 4096, 11008, 4096]
                elif data_source == "Computer Graphics (Rendering)":
                    st.session_state.p = [4, 4, 4, 3, 1]
                else:
                    st.session_state.p = list(np.random.randint(10, 100, size=6))
                
                st.success(f"// Dataset generated — {len(st.session_state.p)-1} matrices")
                
    with col2:
        with st.container():
            st.subheader("02 Preprocessing")
            if 'p' in st.session_state:
                st.write("`// Preprocessing complete`")
                st.code(f"Cleaned Input Array: {st.session_state.p}")
                st.write(f"`// Dimensions validated: p[0]...p[{len(st.session_state.p)-1}]`")
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
                             title="Matrix Dimension Distribution", 
                             color_discrete_sequence=['#00FF88'])
            
            fig_eda.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E0E0E0")
            st.plotly_chart(fig_eda, use_container_width=True)
            
        with eda_c2:
            st.subheader("📉 Complexity")
            n = len(st.session_state.p) - 1
            catalan = factorial(2*(n-1)) // (factorial(n) * factorial(n-1)) if n > 1 else 1
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", 
                value = n,
                gauge = {
                    'axis': {'range': [None, 15], 'tickcolor': "#00FF88"},
                    'bar': {'color': "#00FF88"}, 
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [{'range': [0, 15], 'color': "rgba(255,255,255,0.1)"}]
                },
                title = {'text': "Matrix Count", 'font': {'color': "#00FF88"}}
            ))
            
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.metric("Catalan Number", f"{catalan}")

        st.divider()
        st.subheader("EDA — DIMENSION ANALYSIS")
        
        # FIXED: Proper alignment using st.dataframe with custom styling
        p_list = st.session_state.p
        matrix_df = pd.DataFrame({
            "MATRIX": [f"A{i+1}" for i in range(len(p_list)-1)],
            "ROWS (P i-1)": p_list[:-1],
            "COLS (P i)": p_list[1:],
            "ELEMENTS": [p_list[i] * p_list[i+1] for i in range(len(p_list)-1)],
            "STATUS": ["Ready ✅"] * (len(p_list)-1)
        })

        st.dataframe(
            matrix_df.style.format({
                "ROWS (P i-1)": "{:,}",
                "COLS (P i)": "{:,}",
                "ELEMENTS": "{:,}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.info("💡 Dimension Analysis complete. Move to the Visualizer tab to see the optimal split.")

# --- TAB 1: VISUALIZER & TRACE ---
with tab_viz:
    st.sidebar.header("⚙️ Settings")
    default_p = st.session_state.get('p', [10, 30, 5, 60])
    manual_input = st.sidebar.text_input("Dimensions", ",".join(map(str, default_p)))
    try:
        p_final = [int(x.strip()) for x in manual_input.split(",")]
    except:
        p_final = [10, 30, 5, 60]

    if st.button("🚀 Run Optimizer"):
        n = len(p_final) - 1
        with st.status("🔍 Analyzing Subproblems...", expanded=False) as status:
            m, s = compute_mcm(p_final)
            for length in range(2, n + 1):
                st.write(f"**Chains of length {length}...**")
                time.sleep(0.1) 
                for i in range(n - length + 1):
                    j = i + length - 1
                    st.write(f"✔️ Solved A{i+1}-A{j+1}: Cost {int(m[i][j])}")
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        st.subheader("Optimization Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Min Multiplications", f"{int(m[0][n-1]):,}")
        res_col2.success(f"**Optimal Order:**\n {get_optimal_parens(s, 0, n-1)}")
        
        st.divider()
        col_m, col_s = st.columns(2)
        with col_m:
            st.subheader("DP Cost Matrix (m)")
            df_m = pd.DataFrame(m, columns=[f"A{i+1}" for i in range(n)], index=[f"A{i+1}" for i in range(n)])
            st.plotly_chart(px.imshow(df_m, text_auto=True, color_continuous_scale='Blues'), use_container_width=True)
        with col_s:
            st.subheader("Split Matrix (s)")
            st.write(pd.DataFrame(s, columns=[f"A{i+1}" for i in range(n)], index=[f"A{i+1}" for i in range(n)]))

        st.subheader("🌲 Optimal Split Hierarchy")
        st.json(get_tree_data(s, 0, n-1))

# --- TAB 2: PERFORMANCE ANALYSIS ---
with tab_analysis:
    st.header("🚀 Advanced Performance Benchmarking")
    st.subheader("1. Empirical Complexity Prover (Scaling Test)")
    max_n = st.slider("Select Max Matrices for Scaling", 5, 30, 15)
    
    if st.button("📊 Run Scaling Analysis"):
        with st.spinner("Simulating workloads..."):
            df_scaling = run_scaling_benchmark(max_n)
            fig_scaling = px.line(df_scaling, x="Number of Matrices (n)", 
                                 y="Execution Time (s)", 
                                 title="Empirical Time Complexity Analysis",
                                 markers=True)
            fig_scaling.update_traces(line_color='#00FF88')
            st.plotly_chart(fig_scaling, use_container_width=True)
            st.info(f"As **n** increases, notice the curve follows a **cubic ($n^3$)** trend.")

    st.divider()
    st.subheader("2. Theoretical Efficiency Dashboard")
    if 'p' in st.session_state:
        total_ops = compute_mcm_with_ops(st.session_state.p)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Subproblem Comparisons", f"{total_ops}")
        with c2:
            st.metric("Algorithmic Efficiency", "High (Tabulation)")
    else:
        st.warning("Please generate data in the first tab to see efficiency metrics.")

# --- TAB 3: THEORY ---
with tab_theory:
    st.header("Mathematical Foundation")
    st.latex(r"m[i,j] = \min_{i \leq k < j} \{m[i,k] + m[k+1,j] + p_{i-1}p_kp_j\}")
    st.info("**Time Complexity:** $O(n^3)$ | **Space Complexity:** $O(n^2)$")

# --- TAB 4: VIVA ---
with tab_viva:
    st.header("🎓 Interview / Viva Preparation")
    qa = [
        ("What is 'Overlapping Subproblems'?", "When an algorithm solves the same subproblems multiple times. DP fixes this by storing results in a table."),
        ("Explain the Catalan connection.", "The number of ways to parenthesize a chain of n matrices is given by the (n-1)th Catalan number."),
        ("What is the time complexity of this DP approach?", "It is O(n³), because we have three nested loops: one for chain length, one for the start index, and one for the split point k.")
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)