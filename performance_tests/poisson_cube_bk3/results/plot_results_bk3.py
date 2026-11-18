import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Your raw data string
data_1_gpu = """cells    dofs   mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256    18513    6.338e-05 6.321e-05 3.564e-02 7      4.083e-02    
512    35937    6.490e-05 6.502e-05 4.241e-02 7      3.992e-02    
1024   70785    6.943e-05 6.949e-05 4.302e-02 7      4.132e-02    
2048   139425   8.252e-05 8.241e-05 4.519e-02 7      4.367e-02    
4096   274625   9.450e-05 9.427e-05 5.006e-02 7      4.386e-02    
8192   545025   1.307e-04 1.305e-04 5.504e-02 7      4.379e-02    
16384  1081665  2.045e-04 2.044e-04 6.428e-02 7      4.464e-02    
32768  2146689  3.546e-04 3.545e-04 9.031e-02 7      4.506e-02    
65536  4276737  6.554e-04 6.613e-04 1.329e-01 7      4.503e-02    
131072 8520321  1.278e-03 1.282e-03 2.175e-01 7      4.549e-02    
262144 16974593 2.459e-03 2.525e-03 3.897e-01 7      4.562e-02    
524288 33883137 4.937e-03 4.975e-03 7.247e-01 7      4.553e-02 
"""

data_2_gpu = """cells    dofs   mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256     18513    1.635e-04 1.638e-04 6.880e-02 7      4.083e-02    
512     35937    1.664e-04 1.663e-04 7.748e-02 7      3.992e-02    
1024    70785    1.675e-04 1.671e-04 7.758e-02 7      4.132e-02    
2048    139425   1.726e-04 1.733e-04 8.319e-02 7      4.367e-02    
4096    274625   1.887e-04 1.885e-04 9.338e-02 7      4.386e-02    
8192    545025   2.118e-04 2.034e-04 9.696e-02 7      4.379e-02    
16384   1081665  2.403e-04 2.357e-04 1.007e-01 7      4.464e-02    
32768   2146689  3.183e-04 3.185e-04 1.169e-01 7      4.506e-02    
65536   4276737  4.691e-04 4.688e-04 1.378e-01 7      4.503e-02    
131072  8520321  7.888e-04 7.902e-04 1.870e-01 7      4.549e-02    
262144  16974593 1.428e-03 1.433e-03 2.835e-01 7      4.562e-02    
524288  33883137 2.665e-03 2.679e-03 4.511e-01 7      4.553e-02    
1048576 67634433 5.237e-03 5.250e-03 7.996e-01 7      4.584e-02 
"""


data_4_gpu = """
cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256     18513     2.407e-04 2.409e-04 8.751e-02 7      4.083e-02    
512     35937     2.416e-04 2.417e-04 9.845e-02 7      3.992e-02    
1024    70785     2.484e-04 2.485e-04 1.002e-01 7      4.132e-02    
2048    139425    2.460e-04 2.513e-04 1.057e-01 7      4.367e-02    
4096    274625    2.535e-04 2.572e-04 1.165e-01 7      4.386e-02    
8192    545025    2.709e-04 2.706e-04 1.213e-01 7      4.379e-02    
16384   1081665   2.958e-04 2.830e-04 1.291e-01 7      4.464e-02    
32768   2146689   3.217e-04 3.195e-04 1.368e-01 7      4.506e-02    
65536   4276737   4.002e-04 3.988e-04 1.493e-01 7      4.503e-02    
131072  8520321   5.496e-04 5.502e-04 1.738e-01 7      4.549e-02    
262144  16974593  8.725e-04 8.718e-04 2.313e-01 7      4.562e-02    
524288  33883137  1.520e-03 1.526e-03 3.212e-01 7      4.553e-02    
1048576 67634433  2.783e-03 2.800e-03 4.954e-01 7      4.584e-02    
2097152 135005697 5.416e-03 5.428e-03 8.530e-01 7      4.567e-02 
"""

data_8_gpu = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256     18513     3.771e-04 3.766e-04 1.173e-01 7      4.083e-02    
512     35937     3.783e-04 3.777e-04 1.284e-01 7      3.992e-02    
1024    70785     3.785e-04 3.785e-04 1.350e-01 7      4.132e-02    
2048    139425    3.805e-04 3.803e-04 1.411e-01 7      4.367e-02    
4096    274625    3.835e-04 3.835e-04 1.527e-01 7      4.386e-02    
8192    545025    3.875e-04 3.870e-04 1.605e-01 7      4.379e-02    
16384   1081665   4.054e-04 4.048e-04 1.685e-01 7      4.464e-02    
32768   2146689   4.449e-04 4.465e-04 1.841e-01 7      4.506e-02    
65536   4276737   4.667e-04 4.473e-04 1.893e-01 7      4.503e-02    
131072  8520321   5.366e-04 5.355e-04 2.037e-01 7      4.549e-02    
262144  16974593  7.558e-04 7.115e-04 2.369e-01 7      4.562e-02    
524288  33883137  1.020e-03 1.019e-03 2.870e-01 7      4.553e-02    
1048576 67634433  1.690e-03 1.692e-03 3.829e-01 7      4.584e-02    
2097152 135005697 3.036e-03 3.046e-03 5.731e-01 7      4.567e-02    
4194304 269748225 5.642e-03 5.644e-03 9.203e-01 7      4.569e-02 
"""

data_16_gpu = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256     18513     5.082e-04 5.077e-04 1.451e-01 7      4.083e-02    
512     35937     5.092e-04 5.092e-04 1.564e-01 7      3.992e-02    
1024    70785     5.103e-04 5.123e-04 1.637e-01 7      4.132e-02    
2048    139425    5.144e-04 5.102e-04 1.748e-01 7      4.367e-02    
4096    274625    5.139e-04 5.174e-04 1.872e-01 7      4.386e-02    
8192    545025    5.177e-04 5.204e-04 1.958e-01 7      4.379e-02    
16384   1081665   5.272e-04 5.270e-04 2.072e-01 7      4.464e-02    
32768   2146689   5.509e-04 5.504e-04 2.223e-01 7      4.506e-02    
65536   4276737   5.912e-04 5.919e-04 2.332e-01 7      4.503e-02    
131072  8520321   6.552e-04 6.556e-04 2.385e-01 7      4.549e-02    
262144  16974593  8.031e-04 7.580e-04 2.634e-01 7      4.562e-02    
524288  33883137  1.025e-03 9.178e-04 3.085e-01 7      4.553e-02    
1048576 67634433  1.265e-03 1.263e-03 3.543e-01 7      4.584e-02    
2097152 135005697 2.036e-03 2.037e-03 4.643e-01 7      4.567e-02    
4194304 269748225 3.370e-03 3.376e-03 6.504e-01 7      4.569e-02    
8388608 538970625 6.071e-03 6.083e-03 1.022e+00 7      4.601e-02
"""

data_32_gpu = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256      18513      7.009e-04 7.011e-04 1.855e-01 7      4.083e-02    
512      35937      7.030e-04 7.017e-04 1.972e-01 7      3.992e-02    
1024     70785      7.036e-04 7.035e-04 2.053e-01 7      4.132e-02    
2048     139425     7.032e-04 7.038e-04 2.165e-01 7      4.367e-02    
4096     274625     7.064e-04 7.055e-04 2.353e-01 7      4.386e-02    
8192     545025     7.070e-04 7.083e-04 2.438e-01 7      4.379e-02    
16384    1081665    7.160e-04 7.150e-04 2.557e-01 7      4.464e-02    
32768    2146689    7.227e-04 7.241e-04 2.768e-01 7      4.506e-02    
65536    4276737    7.474e-04 7.473e-04 2.888e-01 7      4.503e-02    
131072   8520321    8.000e-04 8.003e-04 3.049e-01 7      4.549e-02    
262144   16974593   8.806e-04 8.809e-04 3.191e-01 7      4.562e-02    
524288   33883137   1.024e-03 1.014e-03 3.343e-01 7      4.553e-02    
1048576  67634433   1.274e-03 1.138e-03 4.024e-01 7      4.584e-02    
2097152  135005697  1.556e-03 1.553e-03 4.830e-01 7      4.567e-02    
4194304  269748225  2.319e-03 2.318e-03 5.515e-01 7      4.569e-02    
8388608  538970625  3.755e-03 3.757e-03 7.490e-01 7      4.601e-02    
16777216 1076890625 6.664e-03 6.680e-03 1.147e+00 7      4.572e-02
"""

data_64_gpu = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256      18513      6.686e-04 6.697e-04 1.774e-01 7      4.083e-02    
512      35937      9.794e-04 9.800e-04 2.569e-01 7      3.992e-02    
1024     70785      9.822e-04 9.815e-04 2.636e-01 7      4.132e-02    
2048     139425     9.822e-04 9.826e-04 2.749e-01 7      4.367e-02    
4096     274625     9.843e-04 9.831e-04 2.946e-01 7      4.386e-02    
8192     545025     9.848e-04 9.861e-04 3.149e-01 7      4.379e-02    
16384    1081665    9.856e-04 9.900e-04 3.257e-01 7      4.464e-02    
32768    2146689    9.895e-04 9.899e-04 3.453e-01 7      4.506e-02    
65536    4276737    9.975e-04 9.968e-04 3.671e-01 7      4.503e-02    
131072   8520321    1.019e-03 1.020e-03 3.836e-01 7      4.549e-02    
262144   16974593   1.083e-03 1.082e-03 4.062e-01 7      4.562e-02    
524288   33883137   1.162e-03 1.163e-03 4.359e-01 7      4.553e-02    
1048576  67634433   1.316e-03 1.317e-03 4.690e-01 7      4.584e-02    
2097152  135005697  1.561e-03 1.507e-03 5.200e-01 7      4.567e-02    
4194304  269748225  2.103e-03 1.855e-03 6.291e-01 7      4.569e-02    
8388608  538970625  2.679e-03 2.681e-03 6.839e-01 7      4.601e-02    
16777216 1076890625 4.224e-03 4.226e-03 8.860e-01 7      4.572e-02    
33554432 2152730625 7.130e-03 7.141e-03 1.286e+00 7      4.591e-02
"""

data_128_gpu = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
256      18513      6.471e-04 6.462e-04 1.724e-01 7      4.083e-02    
512      35937      9.443e-04 9.453e-04 2.478e-01 7      3.992e-02    
1024     70785      9.849e-04 9.847e-04 2.649e-01 7      4.132e-02    
2048     139425     9.846e-04 9.852e-04 2.771e-01 7      4.367e-02    
4096     274625     9.870e-04 9.852e-04 3.000e-01 7      4.386e-02    
8192     545025     9.890e-04 9.870e-04 3.157e-01 7      4.379e-02    
16384    1081665    9.893e-04 9.869e-04 3.273e-01 7      4.464e-02    
32768    2146689    9.942e-04 9.915e-04 3.468e-01 7      4.506e-02    
65536    4276737    9.967e-04 9.940e-04 3.682e-01 7      4.503e-02    
131072   8520321    1.006e-03 1.004e-03 3.813e-01 7      4.549e-02    
262144   16974593   1.025e-03 1.026e-03 4.073e-01 7      4.562e-02    
524288   33883137   1.091e-03 1.093e-03 4.306e-01 7      4.553e-02    
1048576  67634433   1.171e-03 1.172e-03 4.554e-01 7      4.584e-02    
2097152  135005697  1.318e-03 1.318e-03 5.133e-01 7      4.567e-02    
4194304  269748225  1.599e-03 1.583e-03 5.426e-01 7      4.569e-02    
8388608  538970625  1.877e-03 1.860e-03 6.165e-01 7      4.601e-02    
16777216 1076890625 2.685e-03 2.687e-03 7.194e-01 7      4.572e-02    
33554432 2152730625 4.228e-03 4.227e-03 9.411e-01 7      4.591e-02    
67108864 4303361025 7.142e-03 7.145e-03 1.353e+00 7      4.639e-02
"""


gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128]


def parse_solver_table(raw_string, gpus):
    cleaned_data = raw_string.replace(':', ' ')

    df = pd.read_csv(io.StringIO(cleaned_data), sep=r'\s+')
    df['gpus'] = gpus
    return df


def plot_performance_cg_time(dataframes, y_column='cg_time'):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000]

    fig, ax = plt.subplots()

    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')

        if len(group) >= 1:
            ax.plot(group['gpus'], group[y_column],
                    marker='o', label=f'DOFs: {dof:,}')

    ax.set_xscale('log', base=2)
    plt.yscale('symlog', base=2, linthresh=0.1)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xticks(gpu_counts)
    ax.set_xticklabels([str(g) for g in gpu_counts])

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (s)')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.set_title("CG time")

    plt.legend(
        title="Problem Size",
        fontsize=12,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        borderaxespad=0,
        ncol=3)


def plot_performance_matvec(dataframes, y_column='mv_outer'):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000]

    fig, ax = plt.subplots()

    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')

        if len(group) >= 1:
            ax.plot(group['gpus'], group[y_column]*1000,
                    marker='o', label=f'DOFs: {dof:,}')

    ax.set_xscale('log', base=2)
    plt.yscale('symlog', base=2, linthresh=0.1)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xticks(gpu_counts)
    ax.set_xticklabels([str(g) for g in gpu_counts])

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.set_title("Matvec time")

    plt.legend(
        title="Problem Size",
        fontsize=12,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        borderaxespad=0,
        ncol=3)

def plot_parallel_efficiency(dataframes):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000].copy()

    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')

        t_ref = group['mv_outer'].iloc[0]
        p_ref = group['gpus'].iloc[0]

        efficiency = (t_ref / group['mv_outer']) / (group['gpus'] / p_ref)

        print(f"efficiency ({dof} dofs): {efficiency}")
        plt.plot(group['gpus'], efficiency * 100,
                 marker='s', label=f'DOFs: {dof:,}')

    # Formatting
    plt.xscale('log', base=2)

    plt.xticks(gpu_counts, labels=[str(g) for g in gpu_counts])
    plt.ylim(0, 110)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Parallel Efficiency (%)')
    plt.legend(title='Problem Size')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_parallel_efficiency(combined_df)


df_1 = parse_solver_table(data_1_gpu, 1)
df_2 = parse_solver_table(data_2_gpu, 2)
df_4 = parse_solver_table(data_4_gpu, 4)
df_8 = parse_solver_table(data_8_gpu, 8)
df_16 = parse_solver_table(data_16_gpu, 16)
df_32 = parse_solver_table(data_32_gpu, 32)
df_64 = parse_solver_table(data_64_gpu, 64)
df_128 = parse_solver_table(data_128_gpu, 128)


df_list = [df_1, df_2, df_4, df_8, df_16, df_32, df_64, df_128]
# print(df_1_node)
# print(df_2_node)
# print(df_4_node)
# print(df_8_node)


plot_performance_cg_time(
    df_list)
# plt.figure()
plot_performance_matvec(
    df_list)
plt.show()

# plot_parallel_efficiency(
#     df_list)
