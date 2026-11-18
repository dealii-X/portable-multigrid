import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, LogFormatterSciNotation, FixedLocator, FixedFormatter
import numpy as np
import matplotlib.ticker as ticker


# Your raw data string
data_1_node = """
    cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
0:     256     18513 2.564e-04 2.561e-04 8.632e-02      7    4.083e-02 
0:     512     35937 2.579e-04 2.576e-04 9.721e-02      7    3.992e-02 
0:    1024     70785 2.588e-04 2.575e-04 9.904e-02      7    4.132e-02 
0:    2048    139425 2.624e-04 2.589e-04 1.046e-01      7    4.367e-02 
0:    4096    274625 3.020e-04 2.703e-04 1.179e-01      7    4.386e-02 
0:    8192    545025 3.163e-04 3.061e-04 1.257e-01      7    4.379e-02 
0:   16384   1081665 3.929e-04 3.927e-04 1.415e-01      7    4.464e-02 
0:   32768   2146689 5.394e-04 5.395e-04 1.568e-01      7    4.506e-02 
0:   65536   4276737 8.291e-04 8.300e-04 1.910e-01      7    4.503e-02 
0:  131072   8520321 1.399e-03 1.399e-03 2.640e-01      7    4.549e-02 
0:  262144  16974593 2.559e-03 2.559e-03 4.152e-01      7    4.562e-02 
0:  524288  33883137 4.840e-03 4.838e-03 6.881e-01      7    4.553e-02 
0: 1048576  67634433 9.363e-03 9.362e-03 1.233e+00      7    4.584e-02 
0: 2097152 135005697 1.846e-02 1.846e-02 2.330e+00      7    4.567e-02 
"""

data_2_node = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
0:     256     18513 3.913e-04 3.916e-04 1.163e-01      7    4.083e-02 
0:     512     35937 3.921e-04 3.923e-04 1.267e-01      7    3.992e-02 
0:    1024     70785 3.941e-04 3.947e-04 1.343e-01      7    4.132e-02 
0:    2048    139425 3.944e-04 3.949e-04 1.395e-01      7    4.367e-02 
0:    4096    274625 3.999e-04 3.988e-04 1.512e-01      7    4.386e-02 
0:    8192    545025 4.338e-04 4.292e-04 1.622e-01      7    4.379e-02 
0:   16384   1081665 4.571e-04 4.390e-04 1.745e-01      7    4.464e-02 
0:   32768   2146689 5.406e-04 5.308e-04 1.949e-01      7    4.506e-02 
0:   65536   4276737 6.731e-04 6.733e-04 2.161e-01      7    4.503e-02 
0:  131072   8520321 9.657e-04 9.662e-04 2.463e-01      7    4.549e-02 
0:  262144  16974593 1.560e-03 1.559e-03 3.265e-01      7    4.562e-02 
0:  524288  33883137 2.699e-03 2.699e-03 4.693e-01      7    4.553e-02 
0: 1048576  67634433 5.007e-03 5.007e-03 7.500e-01      7    4.584e-02 
0: 2097152 135005697 9.617e-03 9.617e-03 1.308e+00      7    4.567e-02 
0: 4194304 269748225 1.865e-02 1.865e-02 2.398e+00      7    4.569e-02 
"""


data_4_node = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
 0:     256     18513 5.203e-04 5.198e-04 1.437e-01      7    4.083e-02 
 0:     512     35937 5.216e-04 5.227e-04 1.551e-01      7    3.992e-02 
 0:    1024     70785 5.233e-04 5.234e-04 1.623e-01      7    4.132e-02 
 0:    2048    139425 5.247e-04 5.245e-04 1.733e-01      7    4.367e-02 
 0:    4096    274625 5.277e-04 5.268e-04 1.846e-01      7    4.386e-02 
 0:    8192    545025 5.325e-04 5.320e-04 1.928e-01      7    4.379e-02 
 0:   16384   1081665 5.758e-04 5.759e-04 2.084e-01      7    4.464e-02 
 0:   32768   2146689 6.254e-04 6.222e-04 2.268e-01      7    4.506e-02 
 0:   65536   4276737 6.685e-04 6.694e-04 2.444e-01      7    4.503e-02 
 0:  131072   8520321 8.278e-04 8.272e-04 2.584e-01      7    4.549e-02 
 0:  262144  16974593 1.153e-03 1.154e-03 3.072e-01      7    4.562e-02 
 0:  524288  33883137 1.750e-03 1.748e-03 3.821e-01      7    4.553e-02 
 0: 1048576  67634433 2.939e-03 2.938e-03 5.339e-01      7    4.584e-02 
 0: 2097152 135005697 5.358e-03 5.361e-03 8.310e-01      7    4.567e-02 
 0: 4194304 269748225 9.992e-03 9.998e-03 1.387e+00      7    4.569e-02 
 0: 8388608 538970625 1.924e-02 1.923e-02 2.515e+00      7    4.601e-02 
"""

data_8_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
 0:      256      18513 7.072e-04 7.069e-04 1.827e-01      7    4.083e-02 
 0:      512      35937 7.104e-04 7.094e-04 1.938e-01      7    3.992e-02 
 0:     1024      70785 7.104e-04 7.104e-04 2.008e-01      7    4.132e-02 
 0:     2048     139425 7.107e-04 7.111e-04 2.119e-01      7    4.367e-02 
 0:     4096     274625 7.145e-04 7.135e-04 2.306e-01      7    4.386e-02 
 0:     8192     545025 7.154e-04 7.157e-04 2.391e-01      7    4.379e-02 
 0:    16384    1081665 7.197e-04 7.198e-04 2.504e-01      7    4.464e-02 
 0:    32768    2146689 7.650e-04 7.655e-04 2.738e-01      7    4.506e-02 
 0:    65536    4276737 8.254e-04 8.254e-04 2.884e-01      7    4.503e-02 
 0:   131072    8520321 8.803e-04 8.700e-04 3.096e-01      7    4.549e-02 
 0:   262144   16974593 1.123e-03 1.030e-03 3.459e-01      7    4.562e-02 
 0:   524288   33883137 1.368e-03 1.354e-03 3.736e-01      7    4.553e-02 
 0:  1048576   67634433 1.983e-03 1.981e-03 4.564e-01      7    4.584e-02 
 0:  2097152  135005697 3.230e-03 3.232e-03 6.201e-01      7    4.567e-02 
 0:  4194304  269748225 5.663e-03 5.664e-03 9.121e-01      7    4.569e-02 
 0:  8388608  538970625 1.038e-02 1.038e-02 1.483e+00      7    4.601e-02 
 0: 16777216 1076890625 1.992e-02 1.990e-02 2.628e+00      7    4.572e-02 
"""

data_16_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
 0:      256      18513 6.877e-04 6.813e-04 1.746e-01      7    4.083e-02 
 0:      512      35937 9.856e-04 9.862e-04 2.552e-01      7    3.992e-02 
 0:     1024      70785 9.943e-04 9.901e-04 2.624e-01      7    4.132e-02 
 0:     2048     139425 9.916e-04 9.932e-04 2.723e-01      7    4.367e-02 
 0:     4096     274625 9.917e-04 9.944e-04 2.911e-01      7    4.386e-02 
 0:     8192     545025 9.942e-04 9.928e-04 3.108e-01      7    4.379e-02 
 0:    16384    1081665 9.973e-04 9.963e-04 3.226e-01      7    4.464e-02 
 0:    32768    2146689 1.001e-03 1.004e-03 3.416e-01      7    4.506e-02 
 0:    65536    4276737 1.045e-03 1.044e-03 3.671e-01      7    4.503e-02 
 0:   131072    8520321 1.106e-03 1.104e-03 3.851e-01      7    4.549e-02 
 0:   262144   16974593 1.228e-03 1.228e-03 4.165e-01      7    4.562e-02 
 0:   524288   33883137 1.390e-03 1.301e-03 4.606e-01      7    4.553e-02 
 0:  1048576   67634433 1.628e-03 1.625e-03 5.207e-01      7    4.584e-02 
 0:  2097152  135005697 2.281e-03 2.281e-03 5.817e-01      7    4.567e-02 
 0:  4194304  269748225 3.530e-03 3.529e-03 7.320e-01      7    4.569e-02 
 0:  8388608  538970625 6.039e-03 6.042e-03 1.035e+00      7    4.601e-02 
 0: 16777216 1076890625 1.088e-02 1.087e-02 1.649e+00      7    4.572e-02 
 0: 33554432 2152730625 2.041e-02 2.053e-02 2.851e+00      7    4.591e-02
 """

data_32_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
  0:      256      18513 6.617e-04 6.859e-04 1.738e-01      7    4.083e-02 
  0:      512      35937 9.778e-04 9.629e-04 2.513e-01      7    3.992e-02 
  0:     1024      70785 1.034e-03 1.015e-03 2.680e-01      7    4.132e-02 
  0:     2048     139425 1.022e-03 1.019e-03 2.800e-01      7    4.367e-02 
  0:     4096     274625 1.019e-03 1.021e-03 2.993e-01      7    4.386e-02 
  0:     8192     545025 1.034e-03 1.061e-03 3.190e-01      7    4.379e-02 
  0:    16384    1081665 1.012e-03 1.007e-03 3.289e-01      7    4.464e-02 
  0:    32768    2146689 1.015e-03 1.020e-03 3.459e-01      7    4.506e-02 
  0:    65536    4276737 1.018e-03 1.008e-03 3.671e-01      7    4.503e-02 
  0:   131072    8520321 1.082e-03 1.099e-03 3.879e-01      7    4.549e-02 
  0:   262144   16974593 1.135e-03 1.135e-03 4.480e-01      7    4.562e-02 
  0:   524288   33883137 1.254e-03 1.266e-03 4.467e-01      7    4.553e-02 
  0:  1048576   67634433 1.470e-03 1.362e-03 5.082e-01      7    4.584e-02 
  0:  2097152  135005697 1.691e-03 1.658e-03 5.517e-01      7    4.567e-02 
  0:  4194304  269748225 2.309e-03 2.291e-03 6.253e-01      7    4.569e-02 
  0:  8388608  538970625 3.544e-03 3.532e-03 7.861e-01      7    4.601e-02 
  0: 16777216 1076890625 6.018e-03 6.033e-03 1.094e+00      7    4.572e-02 
  0: 33554432 2152730625 1.128e-02 1.087e-02 1.700e+00      7    4.591e-02
"""

data_64_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
  0:      256      18513 6.785e-04 6.607e-04 1.744e-01      7    4.083e-02 
  0:      512      35937 9.498e-04 9.474e-04 2.491e-01      7    3.992e-02 
  0:     1024      70785 9.801e-04 9.786e-04 2.929e-01      7    4.132e-02 
  0:     2048     139425 1.018e-03 1.014e-03 2.799e-01      7    4.367e-02 
  0:     4096     274625 1.004e-03 1.015e-03 2.994e-01      7    4.386e-02 
  0:     8192     545025 1.178e-03 1.011e-03 3.150e-01      7    4.379e-02 
  0:    16384    1081665 1.034e-03 1.166e-03 3.593e-01      7    4.464e-02 
  0:    32768    2146689 1.187e-03 1.016e-03 3.463e-01      7    4.506e-02 
  0:    65536    4276737 1.049e-03 1.024e-03 3.993e-01      7    4.503e-02 
  0:   131072    8520321 1.029e-03 1.019e-03 3.791e-01      7    4.549e-02 
  0:   262144   16974593 1.072e-03 1.060e-03 4.345e-01      7    4.562e-02 
  0:   524288   33883137 1.127e-03 1.123e-03 4.304e-01      7    4.553e-02 
  0:  1048576   67634433 1.250e-03 1.247e-03 4.811e-01      7    4.584e-02 
  0:  2097152  135005697 1.461e-03 1.482e-03 5.669e-01      7    4.567e-02 
  0:  4194304  269748225 1.732e-03 1.659e-03 5.990e-01      7    4.569e-02 
  0:  8388608  538970625 2.306e-03 2.299e-03 7.356e-01      7    4.601e-02 
  0: 16777216 1076890625 3.534e-03 3.535e-03 8.228e-01      7    4.572e-02 
  0: 33554432 2152730625 6.532e-03 6.555e-03 1.207e+00      7    4.591e-02
"""


def parse_solver_table(raw_string, nodes):
    cleaned_data = raw_string.replace(':', ' ')

    df = pd.read_csv(io.StringIO(cleaned_data), sep=r'\s+')
    df['gpus'] = nodes * 4
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

    gpu_counts = [4, 8, 16, 32, 64, 128, 256]

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

    # plt.xscale('log', base=2)
    # plt.yscale('log', base=2)
    # # plt.gca().yaxis.set_major_locator( LogLocator(base=10))
    # plt.gca().yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # Locator: base 10, but allow "subs" (multiples) 1 through 9
    # plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=range(1,10)))
    # Formatter: base 10, and labelOnlyBase=False ensures 2x10^1 etc get text
    # plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    # plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(1,10)))
    # plt.gca().yaxis.set_minor_formatter(LogFormatterSciNotation(base=10.0))

    # plt.gca().yaxis.set_minor_formatter(LogFormatterSciNotation(base=10.0, labelOnlyBase=False))

    # y_ticks = np.array([0.5, 1, 2])

    # plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))
    # plt.gca().yaxis.set_major_formatter(FixedFormatter(y_ticks))

    # plt.xlabel('Number of GPUs')
    # plt.ylabel('Time (s)')
    # plt.ylim(0, 4)
    # gpu_counts = [4, 8, 16, 32, 64, 128, 256]
    # plt.xticks(gpu_counts, labels=[str(g) for g in gpu_counts])
    # plt.legend(title="Problem Size")
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.title("CG time")
    # # plt.tight_layout()

    # plt.legend(
    #     title="Problem Size",
    #     fontsize=12,
    #     loc='lower left',
    #     bbox_to_anchor=(0.01, 0.01),  # Moves legend outside to the right
    #     borderaxespad=0,
    #     ncol=3                    # Increase to 2 if you have many DOFs
    # )
    # # plt.show()


def plot_performance_matvec(dataframes, y_column='mv_outer'):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000]

    fig, ax = plt.subplots()

    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')

        if len(group) >= 1:
            ax.plot(group['gpus'], group[y_column]*1000,
                    marker='o', label=f'DOFs: {dof:,}')

    gpu_counts = [4, 8, 16, 32, 64, 128, 256]

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
    gpu_counts = [4, 8, 16, 32, 64, 128]

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


df_1_node = parse_solver_table(data_1_node, 1)
df_2_node = parse_solver_table(data_2_node, 2)
df_4_node = parse_solver_table(data_4_node, 4)
df_8_node = parse_solver_table(data_8_node, 8)
df_16_node = parse_solver_table(data_16_node, 16)
df_32_node = parse_solver_table(data_32_node, 32)
df_64_node = parse_solver_table(data_64_node, 64)


# print(df_1_node)
# print(df_2_node)
# print(df_4_node)
# print(df_8_node)


plot_performance_cg_time(
    [df_1_node, df_2_node, df_4_node, df_8_node, df_16_node, df_32_node, df_64_node])
plt.figure()
plot_performance_matvec(
    [df_1_node, df_2_node, df_4_node, df_8_node, df_16_node, df_32_node, df_64_node])
plt.show()

# plot_parallel_efficiency([df_1_node, df_2_node, df_4_node, df_8_node, df_16_node, df_32_node])
