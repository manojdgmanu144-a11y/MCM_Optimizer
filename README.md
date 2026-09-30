<<<<<<< HEAD
# 🧩 MCM Data Pipeline Pro

A high-performance visualization suite for the **Matrix Chain Multiplication (MCM)** problem. This project treats MCM as a real-world data science pipeline—from data collection and preprocessing to optimal execution and benchmarking.

## ✨ Key Features
- **📈 Data Pipeline & EDA:** Simulate workloads from Deep Learning (LLM layers) and Computer Graphics.
- **📊 Interactive Visualizer:** Real-time generation of DP tables ($m$ and $s$ matrices) using Plotly heatmaps.
- **🔍 Live Trace Status:** A "spinning" execution trace that shows subproblems being solved in real-time.
- **🌲 Split Hierarchy:** Interactive JSON tree visualization of the optimal parenthesization.
- **🚀 Benchmarking:** Comparison between $O(n^3)$ DP and $O(2^n)$ Naive Recursion.
- **🎓 Viva & Theory:** Built-in LaTeX recurrence relations and interview Q&A.

## 🛠️ Project Structure
## 🛠️ Project Structure

```text
MCM_Optimizer/
│
├── app.py                 # Main Streamlit application
├── README.md              # Project documentation
└── src/
    ├── __init__.py
    ├── mcm_logic.py       # Core DP & Recursive algorithms
    └── data_handler.py    # Data cleaning logic