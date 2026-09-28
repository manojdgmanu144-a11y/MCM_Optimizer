import numpy as np
import pandas as pd
import time

def compute_mcm(p):
    n = len(p) - 1
    m = np.zeros((n, n))
    s = np.zeros((n, n))
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m, s

def get_optimal_parens(s, i, j):
    if i == j: return f"A{i+1}"
    k = int(s[i][j])
    return f"({get_optimal_parens(s, i, k)} x {get_optimal_parens(s, k + 1, j)})"

def recursive_mcm(p, i, j):
    if i == j: return 0
    res = float('inf')
    for k in range(i, j):
        count = (recursive_mcm(p, i, k) + recursive_mcm(p, k + 1, j) + p[i] * p[k + 1] * p[j + 1])
        if count < res: res = count
    return res

def get_tree_data(s, i, j):
    if i == j: return {"name": f"Matrix A{i+1}"}
    k = int(s[i][j])
    return {
        "name": f"Split at k={k+1}",
        "children": [get_tree_data(s, i, k), get_tree_data(s, k + 1, j)]
    }

def compute_mcm_with_ops(p):
    n = len(p) - 1
    ops_count = 0
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                ops_count += 1
    return ops_count

def run_scaling_benchmark(max_matrices):
    results = []
    for n in range(2, max_matrices + 1):
        p_test = [10] * (n + 1)
        t0 = time.perf_counter()
        compute_mcm(p_test)
        t1 = time.perf_counter()
        results.append({"Number of Matrices (n)": n, "Execution Time (s)": t1 - t0})
    return pd.DataFrame(results)