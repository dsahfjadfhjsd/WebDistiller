"""
Figure 3: Memory Folding - Nature/Science Style (KDD Version)
=============================================================
2x2 layout with centroid similarity (mu) metric.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.gridspec as gridspec

# Nature/Science style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'grid.linewidth': 0.3,
    'lines.linewidth': 0.8,
    'text.usetex': False,
})

COLORS = {
    'raw': '#4393C3',
    'raw_fill': '#92C5DE',
    'folded': '#D6604D',
    'folded_fill': '#F4A582',
    'accent': '#2166AC',
    'green': '#1B7837',
    'gray': '#666666',
    'lightgray': '#E0E0E0',
    'dark': '#1A1A1A',
}

# Precomputed centroid cosine similarity (from embedding space)
MU_RAW = 0.35
MU_FOLDED = 0.82
# Token counts from actual runs
TOKENS_RAW = 20173
TOKENS_FOLDED = 3582


def load_tsne_data(data_path: str):
    data = np.load(data_path)
    return data['raw_tsne'], data['folded_tsne']


def draw_ellipse(ax, coords, color, fill_color, alpha=0.15):
    mean = coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * 1.8 * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=fill_color, alpha=alpha, edgecolor=color,
                      linewidth=0.8, linestyle='-')
    ax.add_patch(ellipse)


def plot_nature_style(raw_tsne, folded_tsne, output_path):
    fig = plt.figure(figsize=(3.6, 3.5), dpi=300)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1.1, 0.9],
                           wspace=0.38, hspace=0.45,
                           left=0.12, right=0.96, top=0.93, bottom=0.08)

    raw_centroid = raw_tsne.mean(axis=0)
    folded_centroid = folded_tsne.mean(axis=0)

    all_coords = np.vstack([raw_tsne, folded_tsne])
    margin = 2.5
    xlim = (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ylim = (all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)

    # ========== (a) Before folding ==========
    ax1 = fig.add_subplot(gs[0, 0])
    draw_ellipse(ax1, raw_tsne, COLORS['raw'], COLORS['raw_fill'], alpha=0.18)
    ax1.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=14, alpha=0.8,
                edgecolors='white', linewidths=0.3, zorder=3)
    ax1.scatter(*raw_centroid, c=COLORS['dark'], s=35,
                marker='+', linewidths=1.0, zorder=5)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('t-SNE$_1$')
    ax1.set_ylabel('t-SNE$_2$')
    ax1.set_title('(a) Before folding', loc='left', fontsize=7, pad=3)
    ax1.text(0.97, 0.97, f'$n$={len(raw_tsne)}\n$\\mu$={MU_RAW}',
             transform=ax1.transAxes, fontsize=5.5, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=COLORS['gray'], alpha=0.9, linewidth=0.4))

    # ========== (b) After folding ==========
    ax2 = fig.add_subplot(gs[0, 1])
    draw_ellipse(ax2, folded_tsne, COLORS['folded'], COLORS['folded_fill'], alpha=0.18)
    ax2.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=20, alpha=0.85,
                edgecolors='white', linewidths=0.3, marker='s', zorder=3)
    ax2.scatter(*folded_centroid, c=COLORS['dark'], s=35,
                marker='+', linewidths=1.0, zorder=5)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('t-SNE$_1$')
    ax2.set_ylabel('t-SNE$_2$')
    ax2.set_title('(b) After folding', loc='left', fontsize=7, pad=3)
    ax2.text(0.97, 0.97, f'$n$={len(folded_tsne)}\n$\\mu$={MU_FOLDED}',
             transform=ax2.transAxes, fontsize=5.5, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=COLORS['gray'], alpha=0.9, linewidth=0.4))

    # ========== (c) Efficiency gains ==========
    ax3 = fig.add_subplot(gs[1, 0])

    cohesion_gain = MU_FOLDED / MU_RAW
    compression_ratio = TOKENS_RAW / TOKENS_FOLDED

    categories = ['Cohesion\ngain', 'Token\ncompression']
    values = [cohesion_gain, compression_ratio]
    colors_bar = [COLORS['green'], COLORS['accent']]

    x = np.arange(len(categories))
    bars = ax3.bar(x, values, width=0.50, color=colors_bar,
                   edgecolor='white', linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.15,
                 f'{val:.1f}$\\times$', ha='center', va='bottom', fontsize=6.5,
                 fontweight='bold', color=COLORS['dark'])

    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=6)
    ax3.set_ylabel('Factor ($\\times$)')
    ax3.set_ylim(0, max(values) * 1.20)
    ax3.set_title('(c) Efficiency gains', loc='left', fontsize=7, pad=3)
    ax3.axhline(y=1, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ========== (d) Transformation ==========
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
                c=COLORS['raw'], s=12, alpha=0.35, edgecolors='none')
    ax4.scatter(folded_tsne[:, 0], folded_tsne[:, 1],
                c=COLORS['folded'], s=18, alpha=0.9,
                marker='s', edgecolors='white', linewidths=0.25)

    ax4.annotate('', xy=folded_centroid, xytext=raw_centroid,
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['green'],
                                 lw=1.2, mutation_scale=8))

    mid = (raw_centroid + folded_centroid) / 2
    ax4.text(mid[0] + 2.0, mid[1] + 1.2, '$\\Phi$', fontsize=9,
             ha='left', va='center', color=COLORS['green'], fontweight='bold')

    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_xlabel('t-SNE$_1$')
    ax4.set_ylabel('t-SNE$_2$')
    ax4.set_title('(d) Transformation', loc='left', fontsize=7, pad=3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['raw'],
               markersize=4, label='Raw', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['folded'],
               markersize=4, label='Folded', linestyle='None'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
               edgecolor=COLORS['lightgray'], fontsize=5.5, handletextpad=0.3,
               borderpad=0.3, labelspacing=0.2, frameon=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.03)
    plt.savefig(output_path.replace('.png', '.pdf'),
                bbox_inches='tight', facecolor='white', pad_inches=0.03)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path("E:/websearcher/outputs/visualizations")
    data_path = output_dir / "tsne_data.npz"

    raw_tsne, folded_tsne = load_tsne_data(str(data_path))
    print(f"Data: raw={len(raw_tsne)}, folded={len(folded_tsne)}")

    output_path = str(output_dir / "fig3_memory_folding_nature.png")
    plot_nature_style(raw_tsne, folded_tsne, output_path)
    print("Done!")


if __name__ == "__main__":
    main()