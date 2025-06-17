import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# --- Load data ---
csv_file = '../MDL/scores_log.csv'
df = pd.read_csv(csv_file)

# --- Grid configuration ---
m, n = 1,2                  # 2 rows, 4 columns (last column reserved for colorbar)
n_plots = n - 1              # 3 columns for heatmaps
num_plots = 10                # actual number of heatmaps

fig = plt.figure(figsize=(n * 4, m * 4))
gs = gridspec.GridSpec(m, n, width_ratios=[1] * n_plots + [0.1], wspace=0.4, hspace=0.4)

# Create axes for heatmaps
axes = []
for i in range(m * n_plots):
    row, col = divmod(i, n_plots)
    axes.append(plt.subplot(gs[row, col]))

# Shared colorbar axis
cbar_ax = plt.subplot(gs[:, -1])

# Plot heatmaps or blanks
for i_dck in range(m * n_plots):
    ax = axes[i_dck]
    if i_dck < num_plots:
        df_dck = df[df.DECK == i_dck].copy()
        # Replace -1 with a large gamma value
        df_dck.loc[df_dck.Alex_Gamma == -1, 'Alex_Gamma'] = df_dck.Alex_Gamma.max() + 1
        df_dck.loc[df_dck.Bob_Gamma == -1, 'Bob_Gamma'] = df_dck.Bob_Gamma.max() + 1

        # Create heatmap data
        heatmap_df = df_dck.pivot(index='Alex_Gamma', columns='Bob_Gamma', values='Win Rate')
        heatmap_df = heatmap_df.sort_index().sort_index(axis=1)

        # Plot heatmap
        sns.heatmap(
            heatmap_df,
            ax=ax,
            cbar=(i_dck == 0),
            cbar_ax=cbar_ax if i_dck == 0 else None,
            annot=False,
            vmin=0,
            vmax=1,
            cmap='viridis'
        )

        # Set title and tick formatting
        ax.set_title(f"Heatmap {i_dck}", fontsize=7)
        if i_dck != 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', length=0)
        else:
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
    else:
        ax.axis('off')  # Blank subplot

# Save or display
plt.tight_layout()
plt.savefig("all_heatmaps.png", dpi=300)
# plt.show()
