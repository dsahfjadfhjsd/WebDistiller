"""
Enhanced t-SNE Visualization for Memory Folding
================================================
生成论文级别的可视化图表，包括：
1. 带密度等高线的 t-SNE 图
2. 压缩比统计图
3. 分层记忆结构可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path


def load_tsne_data(data_path: str):
    """加载已计算的 t-SNE 数据"""
    data = np.load(data_path)
    return {
        'raw_tsne': data['raw_tsne'],
        'folded_tsne': data['folded_tsne'],
        'raw_embeddings': data['raw_embeddings'],
        'folded_embeddings': data['folded_embeddings']
    }


def plot_tsne_with_density(
    raw_tsne: np.ndarray,
    folded_tsne: np.ndarray,
    output_path: str
):
    """绘制带密度等高线的 t-SNE 图（论文风格）"""

    fig, ax = plt.subplots(figsize=(8, 7))

    # 计算密度等高线（只对 raw 数据，因为它更分散）
    if len(raw_tsne) > 10:
        try:
            # 核密度估计
            xx, yy = np.mgrid[
                raw_tsne[:, 0].min()-2:raw_tsne[:, 0].max()+2:100j,
                raw_tsne[:, 1].min()-2:raw_tsne[:, 1].max()+2:100j
            ]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([raw_tsne[:, 0], raw_tsne[:, 1]])
            kernel = stats.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)

            # 绘制等高线（淡蓝色）
            ax.contourf(xx, yy, f, levels=8, cmap='Blues', alpha=0.3)
            ax.contour(xx, yy, f, levels=5, colors='#3498db', alpha=0.4, linewidths=0.5)
        except Exception as e:
            print(f"Density estimation skipped: {e}")

    # 绘制 Raw Content（蓝色圆点）
    ax.scatter(
        raw_tsne[:, 0], raw_tsne[:, 1],
        c='#3498db', s=70, alpha=0.7,
        label='Raw Web Content',
        edgecolors='white', linewidths=0.5,
        zorder=2
    )

    # 绘制 Folded Memory（红色方块，更大更显眼）
    ax.scatter(
        folded_tsne[:, 0], folded_tsne[:, 1],
        c='#e74c3c', s=120, alpha=0.9,
        label='Folded Memory',
        edgecolors='white', linewidths=0.8,
        marker='s', zorder=3
    )

    # 绘制 Folded Memory 的聚类中心
    center = folded_tsne.mean(axis=0)
    ax.scatter(
        center[0], center[1],
        c='darkred', s=200, marker='*',
        edgecolors='white', linewidths=1,
        label='Memory Centroid', zorder=4
    )

    # 绘制 Folded Memory 的凸包（可选）
    if len(folded_tsne) > 2:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(folded_tsne)
            for simplex in hull.simplices:
                ax.plot(
                    folded_tsne[simplex, 0],
                    folded_tsne[simplex, 1],
                    'r--', alpha=0.5, linewidth=1
                )
        except:
            pass

    # 图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    # 标题和标签
    ax.set_title('Memory Folding: Knowledge Density Visualization',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)

    # 计算统计数据
    raw_spread = np.std(raw_tsne, axis=0).mean()
    folded_spread = np.std(folded_tsne, axis=0).mean()
    density_ratio = raw_spread / folded_spread

    # 添加统计信息框
    stats_text = (
        f'Raw Spread: {raw_spread:.2f}\n'
        f'Folded Spread: {folded_spread:.2f}\n'
        f'Density Gain: {density_ratio:.2f}×'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_compression_statistics(output_path: str):
    """绘制压缩比统计图"""

    # 基于实际数据的统计
    categories = ['Raw Content', 'Folded Memory']
    avg_chars = [20173, 3582]  # 从之前的统计
    colors = ['#3498db', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：字符数对比
    ax1 = axes[0]
    bars = ax1.bar(categories, avg_chars, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Average Characters', fontsize=11)
    ax1.set_title('(a) Content Length Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(avg_chars) * 1.2)

    # 添加数值标签
    for bar, val in zip(bars, avg_chars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加压缩比标注
    ax1.annotate(
        '', xy=(1, avg_chars[1] + 1000), xytext=(0, avg_chars[0] - 1000),
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5)
    )
    ax1.text(0.5, (avg_chars[0] + avg_chars[1])/2 + 2000,
             f'5.6× compression', ha='center', fontsize=10, color='gray')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右图：信息密度对比（基于 t-SNE spread）
    ax2 = axes[1]
    spreads = [6.01, 2.56]  # 从之前的计算
    densities = [1/s for s in spreads]  # 密度是 spread 的倒数

    bars2 = ax2.bar(categories, densities, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Information Density (1/spread)', fontsize=11)
    ax2.set_title('(b) Representation Density', fontsize=12, fontweight='bold')

    for bar, val in zip(bars2, densities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Memory Folding: Compression and Density Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_hierarchical_memory_structure(output_path: str):
    """绘制分层记忆结构示意图"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # 定义三层结构
    layers = [
        ('Episode Memory', 'Task milestones & key discoveries', '#2ecc71', 0.8),
        ('Working Memory', 'Current goals & next actions', '#f39c12', 0.6),
        ('Tool Memory', 'Tool usage patterns & rules', '#9b59b6', 0.4)
    ]

    # 绘制层次结构（从下到上）
    y_positions = [0.2, 0.5, 0.8]

    for i, (name, desc, color, width) in enumerate(layers):
        y = y_positions[i]

        # 绘制矩形框
        rect = mpatches.FancyBboxPatch(
            (0.5 - width/2, y - 0.08), width, 0.16,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.8
        )
        ax.add_patch(rect)

        # 添加层名称
        ax.text(0.5, y + 0.02, name, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax.text(0.5, y - 0.03, desc, ha='center', va='center',
                fontsize=9, color='white', alpha=0.9)

    # 添加箭头连接
    for i in range(len(y_positions) - 1):
        ax.annotate(
            '', xy=(0.5, y_positions[i+1] - 0.08),
            xytext=(0.5, y_positions[i] + 0.08),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2)
        )

    # 添加输入/输出标注
    ax.annotate(
        'Raw Context\n(~20K chars)', xy=(0.1, 0.2), fontsize=10,
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    ax.annotate(
        '', xy=(0.25, 0.2), xytext=(0.15, 0.2),
        arrowprops=dict(arrowstyle='->', color='#3498db', lw=2)
    )

    ax.annotate(
        'Compressed Memory\n(~3.5K chars)', xy=(0.9, 0.5), fontsize=10,
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    )
    ax.annotate(
        '', xy=(0.85, 0.5), xytext=(0.75, 0.5),
        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2)
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Three-Layer Memory Folding Architecture',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_paper_figure(
    raw_tsne: np.ndarray,
    folded_tsne: np.ndarray,
    output_path: str
):
    """生成论文主图（WebThinker Fig.5 风格）"""

    fig = plt.figure(figsize=(14, 5))

    # 三个子图
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)

    # (a) Raw Web Content
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(
        raw_tsne[:, 0], raw_tsne[:, 1],
        c='#3498db', s=50, alpha=0.7,
        edgecolors='white', linewidths=0.3
    )
    ax1.set_title('(a) Raw Web Content', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dimension 1', fontsize=10)
    ax1.set_ylabel('Dimension 2', fontsize=10)
    raw_spread = np.std(raw_tsne, axis=0).mean()
    ax1.text(0.05, 0.95, f'Spread: {raw_spread:.2f}',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.grid(True, alpha=0.3, linestyle='--')

    # (b) Folded Memory
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(
        folded_tsne[:, 0], folded_tsne[:, 1],
        c='#e74c3c', s=80, alpha=0.8,
        edgecolors='white', linewidths=0.5,
        marker='s'
    )
    ax2.set_title('(b) Folded Memory', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dimension 1', fontsize=10)
    ax2.set_ylabel('Dimension 2', fontsize=10)
    folded_spread = np.std(folded_tsne, axis=0).mean()
    ax2.text(0.05, 0.95, f'Spread: {folded_spread:.2f}',
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 统一坐标范围
    all_coords = np.vstack([raw_tsne, folded_tsne])
    margin = 3
    xlim = (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ylim = (all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    # (c) Compression Statistics
    ax3 = fig.add_subplot(gs[2])

    metrics = ['Avg Chars', 'Spread', 'Density']
    raw_vals = [20173/1000, raw_spread, 1/raw_spread]  # 归一化
    folded_vals = [3582/1000, folded_spread, 1/folded_spread]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, raw_vals, width, label='Raw', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, folded_vals, width, label='Folded', color='#e74c3c', alpha=0.8)

    ax3.set_ylabel('Value', fontsize=10)
    ax3.set_title('(c) Compression Metrics', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Chars (K)', 'Spread', 'Density'])
    ax3.legend(fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 添加压缩比标注
    compression_ratio = 20173 / 3582
    density_gain = raw_spread / folded_spread

    ax3.text(0.5, 0.85, f'Compression: {compression_ratio:.1f}×',
             transform=ax3.transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax3.text(0.5, 0.72, f'Density Gain: {density_gain:.1f}×',
             transform=ax3.transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('Memory Folding Effectiveness on GAIA Dataset',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*60)
    print("Enhanced Memory Visualization")
    print("="*60)

    output_dir = Path("E:/websearcher/outputs/visualizations")
    data_path = output_dir / "tsne_data.npz"

    # 加载数据
    print("\nLoading t-SNE data...")
    data = load_tsne_data(str(data_path))

    raw_tsne = data['raw_tsne']
    folded_tsne = data['folded_tsne']

    print(f"Raw samples: {len(raw_tsne)}")
    print(f"Folded samples: {len(folded_tsne)}")

    # 生成各种可视化
    print("\nGenerating visualizations...")

    # 1. 带密度等高线的 t-SNE
    plot_tsne_with_density(
        raw_tsne, folded_tsne,
        str(output_dir / "memory_tsne_density.png")
    )

    # 2. 压缩统计图
    plot_compression_statistics(
        str(output_dir / "memory_compression_stats.png")
    )

    # 3. 分层记忆结构图
    plot_hierarchical_memory_structure(
        str(output_dir / "memory_architecture.png")
    )

    # 4. 论文主图（三合一）
    plot_paper_figure(
        raw_tsne, folded_tsne,
        str(output_dir / "memory_paper_figure.png")
    )

    print("\n" + "="*60)
    print("All visualizations generated!")
    print(f"Output: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
