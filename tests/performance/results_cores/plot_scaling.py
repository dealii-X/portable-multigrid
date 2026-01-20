import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_scaling_analysis():
    core_counts = [1, 2, 4, 8, 16, 21, 64, 128, 256]
    all_data = []

    for cores in core_counts:
        filename = f"{cores}-core-full_statistics.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = df[df['name'] == 'solve2'][['n_dofs', 'mean']].copy()
            df['nodes'] = cores
            df['dof_per_node'] = df['n_dofs'] / df['nodes']
            all_data.append(df)
    
    if not all_data:
        print("No data found!")
        return
        
    master_df = pd.concat(all_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: STRONG SCALING (Improved) ---
    unique_dofs = sorted(master_df['n_dofs'].unique())
    for dof in unique_dofs:
        subset = master_df[master_df['n_dofs'] == dof].sort_values('nodes')
        if len(subset) > 1:
            ax1.plot(subset['nodes'], subset['mean'], marker='o', label=f'Total DOF: {dof}')
    
    ax1.set_xscale('log', base=2)
    # If scaling is poor, Linear Y-axis shows the "flatness" better than Log
    ax1.set_title('Strong Scaling ')
    ax1.set_xlabel('Nodes')
    ax1.set_ylabel('Time (s)')
    ax1.legend(fontsize='x-small', title="Problem Size")
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: PSEUDO-WEAK SCALING ---
    # Since exact matches are rare, we plot Time vs n_dofs 
    # and color them by node count to see the trend.
    for nodes in sorted(master_df['nodes'].unique()):
        subset = master_df[master_df['nodes'] == nodes].sort_values('n_dofs')
        ax2.plot(subset['n_dofs'], subset['mean'], marker='s', label=f'{nodes} Nodes')

    ax2.set_xscale('log')
    ax2.set_title('Throughput: Time vs Problem Size')
    ax2.set_xlabel('Total DOFs')
    ax2.set_ylabel('Time (s)')
    ax2.legend(fontsize='x-small', title="Hardware")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_scaling_analysis()