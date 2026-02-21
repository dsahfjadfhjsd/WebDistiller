"""
Figure 3: Memory Folding Visualization (2x2 Layout for Single Column)
=====================================================================
适合双栏论文单栏宽度的 2x2 布局
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.gridspec as gridspec

# ICML 风格设置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 配色
COLORS = {
    'raw': '#5B8FF9',
    'folded': '#F6685E',
    'accent': '#5AD8A6',
    'bar1': '#5AD8A6',
    'bar2': '#5B8FF9',
    'gray': '#8C8C8C',
    'dark': '#262626',
}


def load_tsne_data(data_path: str):
    data = np.load(data_path)
    return data['raw_tsne'], data['folded_tsne']


def compute_cluster_metrics(coords):
    centroid = coords.mean(axis=0)
    spread = np.std(coords, axis=0).mean()
    return {'centroid': centroid, 'spread': spread}


def draw_confidence_ellipse(ax, coords, color, alpha=0.12, n_std=2.0):
    mean = coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=color, alpha=alpha, edgecolor=color,
                      linewidth=1.0, linestyle='-')
    ax.add_patch(ellipse)


def convex_hull_simple(points):
    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def plot_figure3_2x2(raw_tsne, folded_tsne, output_path):
    """生成 2x2 布局的 Figure 3"""

    # 单栏宽度：约 3.5 inches
    fig = plt.figure(figsize=(3.4, 3.4), dpi=300)
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.4,
                           left=0.12, right=0.95, top=0.92, bottom=0.10)

    raw_metrics = compute_cluster_metrics(raw_tsne)
    folded_metrics = compute_cluster_metrics(folded_tsne)

    all_coords = np.vstack([raw_tsne, folded_tsne])
    margin = 3
    xlim = (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ylim = (all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)

    # ==================== (a) Raw Web Content ====================
    ax1 = fig.add_subplot(gs[0, 0])
    draw_confidence_ellipse(ax1, raw_tsne, COLORS['raw'], alpha=0.10)
    ax1.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=18, alpha=0.7,
                edgecolors='white', linewidths=0.2, zorder=3)
    ax1.scatter(*raw_metrics['centroid'], c=COLORS['dark'], s=50,
                marker='X', edgecolors='white', linewidths=0.6, zorder=5)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('t-SNE Dim 1', fontsize=7)
    ax1.set_ylabel('t-SNE Dim 2', fontsize=7)
    ax1.set_title('(a) Raw Web Content', fontsize=8, pad=4)
    ax1.grid(True, alpha=0.15, linewidth=0.2)

    stats_box = dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor=COLORS['raw'], alpha=0.9, linewidth=0.6)
    ax1.text(0.05, 0.95, f'n={len(raw_tsne)}\n$\\sigma$={raw_metrics["spread"]:.2f}',
             transform=ax1.transAxes, fontsize=6, va='top', ha='left', bbox=stats_box)

    # ==================== (b) Folded Memory ====================
    ax2 = fig.add_subplot(gs[0, 1])
    draw_confidence_ellipse(ax2, folded_tsne, COLORS['folded'], alpha=0.10)

    if len(folded_tsne) > 2:
        try:
            hull_points = convex_hull_simple(folded_tsne)
            hull_points = np.vstack([hull_points, hull_points[0]])
            ax2.fill(hull_points[:, 0], hull_points[:, 1],
                    color=COLORS['folded'], alpha=0.04)
            ax2.plot(hull_points[:, 0], hull_points[:, 1],
                    color=COLORS['folded'], linestyle='--', linewidth=0.6, alpha=0.4)
        except:
            pass

    ax2.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=25, alpha=0.8,
                edgecolors='white', linewidths=0.3, marker='s', zorder=3)
    ax2.scatter(*folded_metrics['centroid'], c=COLORS['dark'], s=60,
                marker='*', edgecolors='white', linewidths=0.6, zorder=5)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('t-SNE Dim 1', fontsize=7)
    ax2.set_ylabel('t-SNE Dim 2', fontsize=7)
    ax2.set_title('(b) Folded Memory', fontsize=8, pad=4)
    ax2.grid(True, alpha=0.15, linewidth=0.2)

    stats_box2 = dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['folded'], alpha=0.9, linewidth=0.6)
    ax2.text(0.05, 0.95, f'n={len(folded_tsne)}\n$\\sigma$={folded_metrics["spread"]:.2f}',
             transform=ax2.transAxes, fontsize=6, va='top', ha='left', bbox=stats_box2)

    # ==================== (c) Efficiency Gains ====================
    ax3 = fig.add_subplot(gs[1, 0])

    density_gain = raw_metrics['spread'] / folded_metrics['spread']
    compression_ratio = 20173 / 3582

    metrics_names = ['Density\nGain', 'Compress.\nRatio']
    metrics_values = [density_gain, compression_ratio]
    bar_colors = [COLORS['bar1'], COLORS['bar2']]

    x_pos = np.arange(len(metrics_names))
    bars = ax3.bar(x_pos, metrics_values, color=bar_colors,
                   edgecolor='white', linewidth=0.8, width=0.5, alpha=0.85)

    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.12,
                f'{val:.1f}$\\times$', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=COLORS['dark'])

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics_names, fontsize=6)
    ax3.set_ylabel('Factor', fontsize=7)
    ax3.set_title('(c) Efficiency Gains', fontsize=8, pad=4)
    ax3.set_ylim(0, max(metrics_values) * 1.22)
    ax3.axhline(y=1, color=COLORS['gray'], linestyle=':', linewidth=0.6, alpha=0.5)

    # ==================== (d) Representation Transition ====================
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=12, alpha=0.35,
                edgecolors='none', label=f'Raw (n={len(raw_tsne)})')
    ax4.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=22, alpha=0.8,
                marker='s', edgecolors='white', linewidths=0.2,
                label=f'Folded (n={len(folded_tsne)})')

    ax4.annotate('', xy=folded_metrics['centroid'], xytext=raw_metrics['centroid'],
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['accent'],
                                lw=1.5, mutation_scale=10))

    mid_point = (raw_metrics['centroid'] + folded_metrics['centroid']) / 2
    ax4.text(mid_point[0] + 1.2, mid_point[1] + 0.8,
             'Folding', fontsize=6, ha='left', va='center',
             color=COLORS['accent'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=COLORS['accent'], alpha=0.85, linewidth=0.5))

    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_xlabel('t-SNE Dim 1', fontsize=7)
    ax4.set_ylabel('t-SNE Dim 2', fontsize=7)
    ax4.set_title('(d) Transition', fontsize=8, pad=4)
    ax4.grid(True, alpha=0.15, linewidth=0.2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['raw'],
               markersize=4, label=f'Raw', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['folded'],
               markersize=4, label=f'Folded', linestyle='None'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
               edgecolor='#CCCCCC', fontsize=5.5, handletextpad=0.2,
               borderpad=0.25, labelspacing=0.2)

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 50)
    print("Generating Figure 3 (2x2 Layout)")
    print("=" * 50)

    output_dir = Path("E:/websearcher/outputs/visualizations")
    data_path = output_dir / "tsne_data.npz"

    raw_tsne, folded_tsne = load_tsne_data(str(data_path))
    print(f"Loaded: raw={len(raw_tsne)}, folded={len(folded_tsne)}")

    output_path = str(output_dir / "fig3_memory_folding_2x2.png")
    plot_figure3_2x2(raw_tsne, folded_tsne, output_path)

    print("=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
