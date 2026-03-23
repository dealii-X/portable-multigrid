import pandas as pd
import io
import matplotlib.pyplot as plt

# Your raw data string
data_1_node = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
0:     256     18513 2.550e-04 2.552e-04 1.518e-01      7    4.083e-02 
0:     512     35937 2.570e-04 2.568e-04 1.290e-01      7    3.992e-02 
0:    1024     70785 2.585e-04 2.571e-04 1.373e-01      7    4.132e-02 
0:    2048    139425 2.619e-04 2.583e-04 1.965e-01      7    4.367e-02 
0:    4096    274625 2.989e-04 2.717e-04 2.162e-01      7    4.386e-02 
0:    8192    545025 3.258e-04 3.056e-04 2.258e-01      7    4.379e-02 
0:   16384   1081665 3.937e-04 3.940e-04 2.841e-01      7    4.464e-02 
0:   32768   2146689 5.422e-04 5.420e-04 3.701e-01      7    4.506e-02 
0:   65536   4276737 8.301e-04 8.303e-04 3.905e-01      7    4.503e-02 
0:  131072   8520321 1.398e-03 1.397e-03 5.115e-01      7    4.549e-02 
0:  262144  16974593 2.558e-03 2.559e-03 7.079e-01      7    4.562e-02 
0:  524288  33883137 4.839e-03 4.844e-03 1.001e+00      7    4.553e-02 
0: 1048576  67634433 9.358e-03 9.358e-03 1.595e+00      7    4.584e-02 
0: 2097152 135005697 1.847e-02 1.847e-02 2.734e+00      7    4.567e-02 
"""

data_2_node = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
0:     256     18513 3.909e-04 3.910e-04 1.825e-01      7    4.083e-02 
0:     512     35937 3.920e-04 3.917e-04 2.086e-01      7    3.992e-02 
0:    1024     70785 3.926e-04 3.928e-04 2.522e-01      7    4.132e-02 
0:    2048    139425 3.948e-04 3.946e-04 2.802e-01      7    4.367e-02 
0:    4096    274625 3.981e-04 3.983e-04 3.271e-01      7    4.386e-02 
0:    8192    545025 4.313e-04 4.270e-04 3.734e-01      7    4.379e-02 
0:   16384   1081665 4.562e-04 4.389e-04 4.284e-01      7    4.464e-02 
0:   32768   2146689 5.360e-04 5.319e-04 4.679e-01      7    4.506e-02 
0:   65536   4276737 6.770e-04 6.768e-04 5.470e-01      7    4.503e-02 
0:  131072   8520321 9.703e-04 9.702e-04 5.929e-01      7    4.549e-02 
0:  262144  16974593 1.561e-03 1.559e-03 7.491e-01      7    4.562e-02 
0:  524288  33883137 2.703e-03 2.703e-03 8.958e-01      7    4.553e-02 
0: 1048576  67634433 5.009e-03 5.005e-03 1.207e+00      7    4.584e-02 
0: 2097152 135005697 9.620e-03 9.620e-03 1.811e+00      7    4.567e-02 
0: 4194304 269748225 1.865e-02 1.865e-02 2.943e+00      7    4.569e-02 
"""


data_4_node = """cells    dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
 0:     256     18513 5.199e-04 5.189e-04 2.129e-01      7    4.083e-02 
 0:     512     35937 5.199e-04 5.187e-04 2.377e-01      7    3.992e-02 
 0:    1024     70785 5.203e-04 5.201e-04 2.983e-01      7    4.132e-02 
 0:    2048    139425 5.207e-04 5.213e-04 3.028e-01      7    4.367e-02 
 0:    4096    274625 5.261e-04 5.255e-04 3.582e-01      7    4.386e-02 
 0:    8192    545025 5.293e-04 5.294e-04 4.267e-01      7    4.379e-02 
 0:   16384   1081665 5.724e-04 5.715e-04 4.524e-01      7    4.464e-02 
 0:   32768   2146689 6.255e-04 6.202e-04 5.178e-01      7    4.506e-02 
 0:   65536   4276737 6.700e-04 6.696e-04 5.872e-01      7    4.503e-02 
 0:  131072   8520321 8.284e-04 8.267e-04 6.095e-01      7    4.549e-02 
 0:  262144  16974593 1.271e-03 1.152e-03 7.213e-01      7    4.562e-02 
 0:  524288  33883137 1.753e-03 1.753e-03 8.782e-01      7    4.553e-02 
 0: 1048576  67634433 2.937e-03 2.938e-03 9.895e-01      7    4.584e-02 
 0: 2097152 135005697 5.363e-03 5.361e-03 1.334e+00      7    4.567e-02 
 0: 4194304 269748225 9.988e-03 9.984e-03 1.951e+00      7    4.569e-02 
 0: 8388608 538970625 1.925e-02 1.925e-02 3.102e+00      7    4.601e-02 
"""

data_8_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
 0:      256      18513 7.153e-04 7.163e-04 2.136e-01      7    4.083e-02 
 0:      512      35937 7.163e-04 7.166e-04 2.707e-01      7    3.992e-02 
 0:     1024      70785 7.132e-04 7.134e-04 3.373e-01      7    4.132e-02 
 0:     2048     139425 7.146e-04 7.152e-04 3.567e-01      7    4.367e-02 
 0:     4096     274625 7.154e-04 7.157e-04 3.773e-01      7    4.386e-02 
 0:     8192     545025 7.177e-04 7.174e-04 4.709e-01      7    4.379e-02 
 0:    16384    1081665 7.216e-04 7.212e-04 4.958e-01      7    4.464e-02 
 0:    32768    2146689 7.676e-04 7.682e-04 5.342e-01      7    4.506e-02 
 0:    65536    4276737 8.282e-04 8.283e-04 6.386e-01      7    4.503e-02 
 0:   131072    8520321 8.842e-04 8.713e-04 6.735e-01      7    4.549e-02 
 0:   262144   16974593 1.153e-03 1.096e-03 7.277e-01      7    4.562e-02 
 0:   524288   33883137 1.385e-03 1.354e-03 8.656e-01      7    4.553e-02 
 0:  1048576   67634433 1.986e-03 1.986e-03 9.813e-01      7    4.584e-02 
 0:  2097152  135005697 3.231e-03 3.229e-03 1.171e+00      7    4.567e-02 
 0:  4194304  269748225 5.654e-03 5.661e-03 1.495e+00      7    4.569e-02 
 0:  8388608  538970625 1.036e-02 1.037e-02 2.083e+00      7    4.601e-02 
 0: 16777216 1076890625 1.988e-02 1.989e-02 3.286e+00      7    4.572e-02 
"""

data_16_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
 0:      256      18513 6.813e-04 6.812e-04 2.004e-01      7    4.083e-02 
 0:      512      35937 9.877e-04 9.886e-04 2.963e-01      7    3.992e-02 
 0:     1024      70785 9.937e-04 9.904e-04 3.935e-01      7    4.132e-02 
 0:     2048     139425 9.899e-04 9.908e-04 4.214e-01      7    4.367e-02 
 0:     4096     274625 9.936e-04 9.921e-04 4.441e-01      7    4.386e-02 
 0:     8192     545025 9.941e-04 9.949e-04 5.112e-01      7    4.379e-02 
 0:    16384    1081665 9.958e-04 9.955e-04 5.667e-01      7    4.464e-02 
 0:    32768    2146689 1.003e-03 1.003e-03 6.041e-01      7    4.506e-02 
 0:    65536    4276737 1.051e-03 1.048e-03 6.911e-01      7    4.503e-02 
 0:   131072    8520321 1.114e-03 1.120e-03 7.473e-01      7    4.549e-02 
 0:   262144   16974593 1.232e-03 1.233e-03 8.058e-01      7    4.562e-02 
 0:   524288   33883137 1.429e-03 1.355e-03 9.023e-01      7    4.553e-02 
 0:  1048576   67634433 1.858e-03 1.627e-03 9.945e-01      7    4.584e-02 
 0:  2097152  135005697 2.301e-03 2.298e-03 1.119e+00      7    4.567e-02 
 0:  4194304  269748225 3.560e-03 3.536e-03 1.380e+00      7    4.569e-02 
 0:  8388608  538970625 6.055e-03 6.054e-03 1.646e+00      7    4.601e-02 
 0: 16777216 1076890625 1.126e-02 1.087e-02 2.264e+00      7    4.572e-02 
 0: 33554432 2152730625 2.035e-02 2.045e-02 3.541e+00      7    4.591e-02
 """

data_32_node = """cells      dofs    mv_outer  mv_inner   cg_time  cg_its cg_reduction 
  0:      256      18513 6.605e-04 6.607e-04 1.977e-01      7    4.083e-02 
  0:      512      35937 9.603e-04 9.706e-04 2.902e-01      7    3.992e-02 
  0:     1024      70785 1.025e-03 1.010e-03 3.551e-01      7    4.132e-02 
  0:     2048     139425 1.003e-03 1.002e-03 4.151e-01      7    4.367e-02 
  0:     4096     274625 9.952e-04 1.000e-03 4.525e-01      7    4.386e-02 
  0:     8192     545025 9.963e-04 9.958e-04 5.299e-01      7    4.379e-02 
  0:    16384    1081665 9.990e-04 9.981e-04 5.400e-01      7    4.464e-02 
  0:    32768    2146689 9.999e-04 1.001e-03 5.953e-01      7    4.506e-02 
  0:    65536    4276737 1.007e-03 1.006e-03 6.838e-01      7    4.503e-02 
  0:   131072    8520321 1.051e-03 1.048e-03 7.198e-01      7    4.549e-02 
  0:   262144   16974593 1.114e-03 1.113e-03 7.910e-01      7    4.562e-02 
  0:   524288   33883137 1.234e-03 1.235e-03 8.817e-01      7    4.553e-02 
  0:  1048576   67634433 1.436e-03 1.342e-03 9.603e-01      7    4.584e-02 
  0:  2097152  135005697 1.691e-03 1.629e-03 1.031e+00      7    4.567e-02 
  0:  4194304  269748225 2.288e-03 2.286e-03 1.209e+00      7    4.569e-02 
  0:  8388608  538970625 3.537e-03 3.532e-03 1.420e+00      7    4.601e-02 
  0: 16777216 1076890625 6.596e-03 6.054e-03 1.761e+00      7    4.572e-02 
  0: 33554432 2152730625 1.090e-02 1.092e-02 2.428e+00      7    4.591e-02
"""


def parse_solver_table(raw_string, nodes):
    cleaned_data = raw_string.replace(':', ' ')

    df = pd.read_csv(io.StringIO(cleaned_data), sep=r'\s+')
    df['gpus'] = nodes * 4
    return df


def plot_performance_cg_time(dataframes, y_column='cg_time'):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000]

    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')

        if len(group) >= 1:
            plt.plot(group['gpus'], group[y_column],
                     marker='o', label=f'DOFs: {dof:,}')

    plt.xscale('log', base=2)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Time (s)')
    plt.ylim(0, 4)
    gpu_counts = [4, 8, 16, 32, 64, 128]
    plt.xticks(gpu_counts, labels=[str(g) for g in gpu_counts])
    plt.legend(title="Problem Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("CG time")
    plt.tight_layout()
    # plt.show()


def plot_performance_matvec(dataframes, y_column='mv_outer'):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000]

    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')

        if len(group) >= 1:
            plt.plot(group['gpus'], group[y_column]*1000,
                     marker='o', label=f'DOFs: {dof:,}')

    plt.xscale('log', base=2)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Time (ms)')
    plt.ylim(0,25)
    gpu_counts = [4, 8, 16, 32, 64, 128]
    
    plt.xticks(gpu_counts, labels=[str(g) for g in gpu_counts])
    plt.legend(title="Problem Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Matvec time")
    plt.tight_layout()
    # plt.show()


def plot_parallel_efficiency(dataframes):

    df = pd.concat(dataframes, ignore_index=True)

    df_filtered = df[df['dofs'] > 10000000].copy()
    
    for dof, group in df_filtered.groupby('dofs'):
        group = group.sort_values('gpus')
        
        t_ref = group['mv_outer'].iloc[0]
        p_ref = group['gpus'].iloc[0]
        
        efficiency = (t_ref / group['mv_outer']) / (group['gpus'] / p_ref)
        
        print(f"efficiency ({dof} dofs): {efficiency}")
        plt.plot(group['gpus'], efficiency * 100, marker='s', label=f'DOFs: {dof:,}')


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


# print(df_1_node)
# print(df_2_node)
# print(df_4_node)
# print(df_8_node)


plot_performance_cg_time(
    [df_1_node, df_2_node, df_4_node, df_8_node, df_16_node, df_32_node])
plt.figure()
plot_performance_matvec(
    [df_1_node, df_2_node, df_4_node, df_8_node, df_16_node, df_32_node])
plt.show()

plot_parallel_efficiency([df_1_node, df_2_node, df_4_node, df_8_node, df_16_node, df_32_node])
