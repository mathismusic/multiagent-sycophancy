def interactive_heatmap():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # 1. Setup Data extracted from the original image
    # Rows: source models, Columns: target models
    # Diagonal values are np.nan (as they are empty in the original image)
    models = ['llama1b', 'llama3b', 'llama8b', 'qwen3b', 'qwen7b', 'qwen14b']

    data_baseline = np.array([
        [np.nan, 65, 45, 37, 78, 42],
        [46, np.nan, 35, 17, 42, 43],
        [54, 62, np.nan, 39, 38, 38],
        [50, 58, 41, np.nan, 42, 41],
        [53, 65, 43, 33, np.nan, 35],
        [51, 61, 46, 33, 30, np.nan]
    ])

    data_bss = np.array([
        [np.nan, 63, 45, 34, 61, 26],
        [79, np.nan, 34, 19, 39, 32],
        [76, 66, np.nan, 33, 36, 26],
        [72, 65, 44, np.nan, 41, 32],
        [74, 71, 44, 31, np.nan, 21],
        [76, 45, 45, 26, 28, np.nan]
    ])

    data_dss = np.array([
        [np.nan, 71, 52, 43, 60, 27],
        [47, np.nan, 44, 19, 34, 37],
        [58, 60, np.nan, 49, 44, 36],
        [50, 55, 59, np.nan, 43, 34],
        [52, 54, 33, 32, np.nan, 27],
        [51, 57, 57, 32, 32, np.nan]
    ])

    # 2. Define the Custom Color Scheme based on your provided code
    # Low Influence = Blue (#4A90E2), High Influence = Red (#E24A4A)
    # We add white in the middle to create a clean diverging map
    colors = ["#4A90E2", "#f7f7f7", "#E24A4A"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors)

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    # Helper function to plot each heatmap
    def plot_heatmap(data, ax, title, show_cbar=False):
        # Create mask for the diagonal (NaN values)
        mask = np.isnan(data)

        sns.heatmap(data, ax=ax, annot=True, fmt='.0f',
                    cmap=custom_cmap,
                    mask=mask,
                    linewidths=1, linecolor='white',
                    cbar=show_cbar, square=True,
                    annot_kws={"size": 12, "weight": "bold"},
                    vmin=20, vmax=80) # Fixing scale to keep colors consistent across plots

        # Customizing axes
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models, rotation=0)
        ax.set_xlabel('Target Model →', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('← Source Model', fontsize=11)
        else:
            ax.set_ylabel('')

    # Plot the three charts
    plot_heatmap(data_baseline, axes[0], "Baseline: Total influence between source and target models")
    plot_heatmap(data_bss, axes[1], "With BSS: Total influence between source and target models")
    plot_heatmap(data_dss, axes[2], "With DSS: Total influence between source and target models", show_cbar=True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig('heatmap_recreation.png', dpi=300, bbox_inches='tight')
    plt.show()


