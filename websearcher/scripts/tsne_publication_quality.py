"""
Publication-Quality t-SNE Visualization for Memory Folding
===========================================================
生成顶会论文级别的可视化图表（NeurIPS/ICML 风格）
不依赖 scipy 高级功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from pathlib import Path
import matplotlib.gridspec as gridspec

# 设置论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'text.usetex': False,
})

# 高级配色方案
COLORS = {
    'raw': '#4A90D9',
    'raw_light': '#A8C8E8',
    'folded': '#E85A5A',
    'folded_light': '#F5B5B5',
    'accent': '#27AE60',
    'gray': '#7F8C8D',
    'dark': '#2C3E50',
}


def load_tsne_data(data_path: str):
    """加载 t-SNE 数据"""
    data = np.load(data_path)
    return data['raw_tsne'], data['folded_tsne']


def compute_cluster_metrics(coords):
    """计算聚类指标"""
    centroid = coords.mean(axis=0)
    distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
    return {
        'centroid': centroid,
        'spread': np.std(coords, axis=0).mean(),
        'cov': np.cov(coords.T),
    }


def draw_confidence_ellipse(ax, coords, color, alpha=0.2, n_std=2.0):
    """绘制置信椭圆"""
    mean = coords.mean(axis=0)
    cov = np.cov(coords.T)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 椭圆参数
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=color, alpha=alpha, edgecolor=color,
                      linewidth=1.5, linestyle='--')
    ax.add_patch(ellipse)
    return ellipse


def convex_hull_simple(points):
    """简单的凸包计算（Graham scan）"""
    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return points

    # 构建下半部分
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 构建上半部分
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])


def plot_publication_figure(raw_tsne, folded_tsne, output_path):
    """生成顶会论文级别的主图"""

    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.2, 1.2, 0.8, 1],
                           wspace=0.28, left=0.05, right=0.98, top=0.88, bottom=0.12)

    raw_metrics = compute_cluster_metrics(raw_tsne)
    folded_metrics = compute_cluster_metrics(folded_tsne)

    all_coords = np.vstack([raw_tsne, folded_tsne])
    margin = 4
    xlim = (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ylim = (all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)

    # ==================== (a) Raw Web Content ====================
    ax1 = fig.add_subplot(gs[0])

    # 散点
    ax1.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=50, alpha=0.7,
                edgecolors='white', linewidths=0.4, zorder=3)

    # 置信椭圆
    draw_confidence_ellipse(ax1, raw_tsne, COLORS['raw'], alpha=0.12, n_std=2.0)

    # 质心
    ax1.scatter(*raw_metrics['centroid'], c=COLORS['dark'], s=100,
                marker='X', edgecolors='white', linewidths=1.2, zorder=5)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.set_title('(a) Raw Web Content', fontweight='bold', pad=8)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.3)

    stats_text = f"$n$={len(raw_tsne)}\n$\\sigma$={raw_metrics['spread']:.2f}"
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=9, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['raw'], alpha=0.95, linewidth=1))

    # ==================== (b) Folded Memory ====================
    ax2 = fig.add_subplot(gs[1])

    ax2.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=75, alpha=0.85,
                edgecolors='white', linewidths=0.5,
                marker='s', zorder=3)

    draw_confidence_ellipse(ax2, folded_tsne, COLORS['folded'], alpha=0.15, n_std=2.0)

    # 凸包
    if len(folded_tsne) > 2:
        try:
            hull_points = convex_hull_simple(folded_tsne)
            hull_points = np.vstack([hull_points, hull_points[0]])
            ax2.plot(hull_points[:, 0], hull_points[:, 1],
                    color=COLORS['folded'], linestyle='--', linewidth=1.2, alpha=0.6)
            ax2.fill(hull_points[:, 0], hull_points[:, 1],
                    color=COLORS['folded'], alpha=0.06)
        except:
            pass

    ax2.scatter(*folded_metrics['centroid'], c=COLORS['dark'], s=130,
                marker='*', edgecolors='white', linewidths=1.2, zorder=5)

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('(b) Folded Memory', fontweight='bold', pad=8)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.3)

    stats_text = f"$n$={len(folded_tsne)}\n$\\sigma$={folded_metrics['spread']:.2f}"
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=9, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['folded'], alpha=0.95, linewidth=1))

    # ==================== (c) Efficiency Gains ====================
    ax3 = fig.add_subplot(gs[2])

    density_gain = raw_metrics['spread'] / folded_metrics['spread']
    compression_ratio = 20173 / 3582

    metrics_names = ['Spread\nReduction', 'Compression\nRatio']
    metrics_values = [density_gain, compression_ratio]
    colors = [COLORS['accent'], COLORS['raw']]

    bars = ax3.bar(metrics_names, metrics_values, color=colors,
                   edgecolor='white', linewidth=1.5, width=0.55, alpha=0.85)

    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.1f}×', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=COLORS['dark'])

    ax3.set_ylabel('Factor (×)', fontsize=10)
    ax3.set_title('(c) Efficiency Gains', fontweight='bold', pad=8)
    ax3.set_ylim(0, max(metrics_values) * 1.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax3.axhline(y=1, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.7)
    ax3.text(1.5, 1.15, 'baseline', fontsize=8, color=COLORS['gray'], ha='right')

    # ==================== (d) Representation Transition ====================
    ax4 = fig.add_subplot(gs[3])

    ax4.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=35, alpha=0.45,
                edgecolors='none', label='Raw Content')

    ax4.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=60, alpha=0.85,
                marker='s', edgecolors='white', linewidths=0.4,
                label='Folded Memory')

    # 压缩方向箭头
    ax4.annotate('', xy=folded_metrics['centroid'], xytext=raw_metrics['centroid'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                                lw=2.5, mutation_scale=15))

    mid_point = (raw_metrics['centroid'] + folded_metrics['centroid']) / 2
    ax4.text(mid_point[0] + 2, mid_point[1] + 1.5,
             'Memory\nFolding', fontsize=9, ha='left', va='center',
             color=COLORS['accent'], fontweight='bold')

    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.set_title('(d) Representation Transition', fontweight='bold', pad=8)
    ax4.grid(True, alpha=0.25, linestyle='-', linewidth=0.3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['raw'],
               markersize=8, label=f'Raw ($n$={len(raw_tsne)})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['folded'],
               markersize=8, label=f'Folded ($n$={len(folded_tsne)})'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
               edgecolor='gray', fontsize=8)

    fig.suptitle('Memory Folding: Knowledge Density Visualization on GAIA Dataset',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_three_layer_architecture(output_path):
    """绘制高级三层记忆架构图"""

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    layer_colors = {
        'episode': '#3498DB',
        'working': '#F39C12',
        'tool': '#9B59B6',
    }

    # 输入框
    input_box = FancyBboxPatch((0.3, 2), 2.2, 2.5,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor='#E8F4FD', edgecolor='#3498DB',
                                linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.4, 4.1, 'Raw Context', fontsize=11, fontweight='bold', ha='center', color='#2C3E50')
    ax.text(1.4, 3.6, '~20K chars', fontsize=9, ha='center', color='#7F8C8D')
    ax.text(1.4, 2.9, 'Web pages\nSearch results\nTool outputs', fontsize=8,
            ha='center', color='#95A5A6', linespacing=1.3)

    # 三层记忆
    layers = [
        ('Episode Memory', 'Task milestones & key facts', layer_colors['episode'], 4.8),
        ('Working Memory', 'Current goals & next actions', layer_colors['working'], 3.25),
        ('Tool Memory', 'Usage patterns & derived rules', layer_colors['tool'], 1.7),
    ]

    for name, desc, color, y in layers:
        box = FancyBboxPatch((4, y - 0.45), 3.5, 1.0,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.88)
        ax.add_patch(box)
        ax.text(5.75, y + 0.15, name, fontsize=10, fontweight='bold', ha='center', color='white')
        ax.text(5.75, y - 0.15, desc, fontsize=8, ha='center', color='white', alpha=0.92)

    # 层间箭头
    for y1, y2 in [(4.35, 3.8), (2.8, 2.25)]:
        ax.annotate('', xy=(5.75, y2), xytext=(5.75, y1),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))

    # 输入箭头
    ax.annotate('', xy=(4, 3.25), xytext=(2.5, 3.25),
                arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2.5))
    ax.text(3.25, 3.55, 'Fold', fontsize=10, ha='center', color='#3498DB', fontweight='bold')

    # 输出框
    output_box = FancyBboxPatch((8.3, 2), 2.2, 2.5,
                                 boxstyle="round,pad=0.05,rounding_size=0.15",
                                 facecolor='#FDEDEC', edgecolor='#E74C3C', linewidth=2)
    ax.add_patch(output_box)
    ax.text(9.4, 4.1, 'Compressed', fontsize=11, fontweight='bold', ha='center', color='#2C3E50')
    ax.text(9.4, 3.6, '~3.5K chars', fontsize=9, ha='center', color='#7F8C8D')
    ax.text(9.4, 2.9, 'Structured JSON\nKey facts\nAction plans', fontsize=8,
            ha='center', color='#95A5A6', linespacing=1.3)

    # 输出箭头
    ax.annotate('', xy=(8.3, 3.25), xytext=(7.5, 3.25),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
    ax.text(7.9, 3.55, 'Output', fontsize=10, ha='center', color='#E74C3C', fontweight='bold')

    # 压缩比
    ax.annotate('', xy=(9.0, 1.2), xytext=(1.8, 1.2),
                arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2))
    ax.text(5.4, 0.85, '5.6× Compression  |  2.4× Density Gain', fontsize=11, ha='center',
            color='#27AE60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#27AE60', alpha=0.95))

    ax.set_title('Three-Layer Hierarchical Memory Folding Architecture',
                 fontsize=13, fontweight='bold', pad=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_ablation_visual(output_path):
    """绘制消融实验可视化"""

    fig, ax = plt.subplots(figsize=(8, 4))

    configs = ['WebSearcher (Full)', 'w/o Hierarchical Memory',
               'w/o Intent-Driven Interaction', 'w/o Dual-Model Orchestration']
    accuracies = [50.5, 43.7, 44.7, 44.7]
    drops = ['—', '−6.8', '−5.8', '−5.8']

    colors = [COLORS['accent']] + [COLORS['raw']] * 3

    bars = ax.barh(configs[::-1], accuracies[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=1.5, height=0.55, alpha=0.85)

    for bar, acc, drop in zip(bars, accuracies[::-1], drops[::-1]):
        width = bar.get_width()
        ax.text(width + 0.8, bar.get_y() + bar.get_height()/2,
                f'{acc}%', va='center', ha='left', fontsize=10, fontweight='bold')
        if drop != '—':
            ax.text(width - 2.5, bar.get_y() + bar.get_height()/2,
                    f'({drop})', va='center', ha='right', fontsize=9,
                    color='white', fontweight='bold')

    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_xlim(0, 60)
    ax.axvline(x=50.5, color=COLORS['accent'], linestyle='--', linewidth=1.5, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Ablation Study on GAIA Dataset', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("Publication-Quality Visualization")
    print("="*60)

    output_dir = Path("E:/websearcher/outputs/visualizations")
    data_path = output_dir / "tsne_data.npz"

    raw_tsne, folded_tsne = load_tsne_data(str(data_path))
    print(f"Loaded: raw={len(raw_tsne)}, folded={len(folded_tsne)}")

    print("\nGenerating publication-quality figures...")

    plot_publication_figure(raw_tsne, folded_tsne,
                           str(output_dir / "fig_memory_tsne_publication.png"))

    plot_three_layer_architecture(str(output_dir / "fig_memory_architecture.png"))

    plot_ablation_visual(str(output_dir / "fig_ablation_study.png"))

    print("\n" + "="*60)
    print("All publication figures generated!")
    print("="*60)


if __name__ == "__main__":
    main()
