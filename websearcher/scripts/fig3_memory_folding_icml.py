"""
Figure 3: Memory Folding Visualization (ICML Publication Style)
===============================================================
高质量的 t-SNE 可视化图，用于展示 Memory Folding 的效果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.gridspec as gridspec

# ICML 2026 风格设置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.6,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 现代配色方案 (适合顶会论文)
COLORS = {
    'raw': '#5B8FF9',        # 柔和蓝色
    'raw_light': '#B8D4FF',
    'folded': '#F6685E',     # 柔和红色
    'folded_light': '#FFCCC7',
    'accent': '#5AD8A6',     # 清新绿色
    'accent2': '#6DC8EC',    # 浅蓝色
    'bar1': '#5AD8A6',       # 绿色条
    'bar2': '#5B8FF9',       # 蓝色条
    'gray': '#8C8C8C',
    'dark': '#262626',
    'bg': '#FAFAFA',
}


def load_tsne_data(data_path: str):
    """加载 t-SNE 数据"""
    data = np.load(data_path)
    return data['raw_tsne'], data['folded_tsne']


def compute_cluster_metrics(coords):
    """计算聚类指标"""
    centroid = coords.mean(axis=0)
    spread = np.std(coords, axis=0).mean()
    return {
        'centroid': centroid,
        'spread': spread,
        'cov': np.cov(coords.T),
    }


def draw_confidence_ellipse(ax, coords, color, alpha=0.15, n_std=2.0):
    """绘制置信椭圆"""
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
                      linewidth=1.2, linestyle='-')
    ax.add_patch(ellipse)
    return ellipse


def convex_hull_simple(points):
    """简单的凸包计算"""
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


def plot_figure3(raw_tsne, folded_tsne, output_path):
    """生成 Figure 3: Memory Folding 可视化"""

    # 创建图形 - 适合双栏论文
    fig = plt.figure(figsize=(7.0, 2.2), dpi=300)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.65, 1.0],
                           wspace=0.35, left=0.06, right=0.98, top=0.82, bottom=0.18)

    raw_metrics = compute_cluster_metrics(raw_tsne)
    folded_metrics = compute_cluster_metrics(folded_tsne)

    # 统一坐标范围
    all_coords = np.vstack([raw_tsne, folded_tsne])
    margin = 3
    xlim = (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ylim = (all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)

    # ==================== (a) Raw Web Content ====================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')

    # 先画置信椭圆（在底层）
    draw_confidence_ellipse(ax1, raw_tsne, COLORS['raw'], alpha=0.12, n_std=2.0)

    # 散点图
    ax1.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=28, alpha=0.75,
                edgecolors='white', linewidths=0.3, zorder=3)

    # 质心标记
    ax1.scatter(*raw_metrics['centroid'], c=COLORS['dark'], s=70,
                marker='X', edgecolors='white', linewidths=0.8, zorder=5)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('t-SNE Dim 1', fontsize=8)
    ax1.set_ylabel('t-SNE Dim 2', fontsize=8)
    ax1.set_title('(a) Raw Web Content', fontsize=9, pad=6)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)

    # 统计信息框
    stats_box = dict(boxstyle='round,pad=0.25', facecolor='white',
                     edgecolor=COLORS['raw'], alpha=0.95, linewidth=0.8)
    ax1.text(0.05, 0.95, f'n={len(raw_tsne)}\n$\\sigma$={raw_metrics["spread"]:.2f}',
             transform=ax1.transAxes, fontsize=7, va='top', ha='left', bbox=stats_box)

    # ==================== (b) Folded Memory ====================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')

    # 置信椭圆
    draw_confidence_ellipse(ax2, folded_tsne, COLORS['folded'], alpha=0.12, n_std=2.0)

    # 凸包
    if len(folded_tsne) > 2:
        try:
            hull_points = convex_hull_simple(folded_tsne)
            hull_points = np.vstack([hull_points, hull_points[0]])
            ax2.fill(hull_points[:, 0], hull_points[:, 1],
                    color=COLORS['folded'], alpha=0.05)
            ax2.plot(hull_points[:, 0], hull_points[:, 1],
                    color=COLORS['folded'], linestyle='--', linewidth=0.8, alpha=0.5)
        except:
            pass

    # 散点图 - 方形标记
    ax2.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=40, alpha=0.85,
                edgecolors='white', linewidths=0.4, marker='s', zorder=3)

    # 质心
    ax2.scatter(*folded_metrics['centroid'], c=COLORS['dark'], s=90,
                marker='*', edgecolors='white', linewidths=0.8, zorder=5)

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('t-SNE Dim 1', fontsize=8)
    ax2.set_ylabel('t-SNE Dim 2', fontsize=8)
    ax2.set_title('(b) Folded Memory', fontsize=9, pad=6)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)

    stats_box2 = dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COLORS['folded'], alpha=0.95, linewidth=0.8)
    ax2.text(0.05, 0.95, f'n={len(folded_tsne)}\n$\\sigma$={folded_metrics["spread"]:.2f}',
             transform=ax2.transAxes, fontsize=7, va='top', ha='left', bbox=stats_box2)

    # ==================== (c) Efficiency Gains ====================
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('white')

    # 计算增益
    density_gain = raw_metrics['spread'] / folded_metrics['spread']
    compression_ratio = 20173 / 3582  # 从原始数据

    metrics_names = ['Density\nGain', 'Compress.\nRatio']
    metrics_values = [density_gain, compression_ratio]
    bar_colors = [COLORS['bar1'], COLORS['bar2']]

    x_pos = np.arange(len(metrics_names))
    bars = ax3.bar(x_pos, metrics_values, color=bar_colors,
                   edgecolor='white', linewidth=1.0, width=0.5, alpha=0.9)

    # 数值标签
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{val:.1f}$\\times$', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=COLORS['dark'])

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics_names, fontsize=7)
    ax3.set_ylabel('Factor', fontsize=8)
    ax3.set_title('(c) Efficiency', fontsize=9, pad=6)
    ax3.set_ylim(0, max(metrics_values) * 1.25)

    # 基准线
    ax3.axhline(y=1, color=COLORS['gray'], linestyle=':', linewidth=0.8, alpha=0.6)

    # ==================== (d) Representation Transition ====================
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor('white')

    # Raw points (较淡)
    ax4.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=20, alpha=0.4,
                edgecolors='none', label=f'Raw (n={len(raw_tsne)})')

    # Folded points (突出)
    ax4.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=35, alpha=0.85,
                marker='s', edgecolors='white', linewidths=0.3,
                label=f'Folded (n={len(folded_tsne)})')

    # 压缩方向箭头
    ax4.annotate('', xy=folded_metrics['centroid'], xytext=raw_metrics['centroid'],
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['accent'],
                                lw=2.0, mutation_scale=12))

    # 标注文字
    mid_point = (raw_metrics['centroid'] + folded_metrics['centroid']) / 2
    offset = np.array([1.5, 1.0])
    ax4.text(mid_point[0] + offset[0], mid_point[1] + offset[1],
             'Memory\nFolding', fontsize=7, ha='left', va='center',
             color=COLORS['accent'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['accent'], alpha=0.9, linewidth=0.6))

    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_xlabel('t-SNE Dim 1', fontsize=8)
    ax4.set_ylabel('t-SNE Dim 2', fontsize=8)
    ax4.set_title('(d) Transition', fontsize=9, pad=6)
    ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['raw'],
               markersize=6, label=f'Raw (n={len(raw_tsne)})', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['folded'],
               markersize=6, label=f'Folded (n={len(folded_tsne)})', linestyle='None'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
               edgecolor='#CCCCCC', fontsize=6.5, handletextpad=0.3,
               borderpad=0.3, labelspacing=0.3)

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.02)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.02)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Figure 3: Memory Folding Visualization")
    print("=" * 60)

    output_dir = Path("E:/websearcher/outputs/visualizations")
    data_path = output_dir / "tsne_data.npz"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    raw_tsne, folded_tsne = load_tsne_data(str(data_path))
    print(f"Loaded: raw={len(raw_tsne)}, folded={len(folded_tsne)}")

    # 生成 Figure 3
    output_path = str(output_dir / "fig3_memory_folding.png")
    plot_figure3(raw_tsne, folded_tsne, output_path)

    print("\n" + "=" * 60)
    print("Figure 3 generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
