"""
Memory Folding t-SNE Visualization
==================================
证明 "认知记忆折叠" 将杂乱的网页噪声转化为紧凑的知识表示

生成图表：效仿 WebThinker 风格的 t-SNE 分布图
- 蓝色点：折叠前的原始内容（Raw Web Content）
- 红色点：折叠后的记忆摘要（Folded Memory）
- 展示两类表示在向量空间中的分布差异
"""

import os
import sys
import json
import re
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']


@dataclass
class TextSample:
    """文本样本"""
    text: str
    category: str  # 'raw_content' or 'folded_memory'
    source_file: str
    sample_id: int
    char_count: int


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_samples_from_file(file_path: str) -> List[TextSample]:
    """从单个结果文件中提取样本"""
    samples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return samples

    file_name = Path(file_path).name

    for item in data:
        output = item.get('Output', '')
        sample_id = item.get('id', 0)
        memory_folds = item.get('Memory_Folds', 0)

        # 提取 tool_call_result（原始内容）
        tool_results = re.findall(
            r'<tool_call_result>(.*?)</tool_call_result>',
            output,
            re.DOTALL
        )

        # 只选取较长的 tool_result（代表真实的网页内容，而非简单的搜索结果）
        for tr in tool_results:
            tr_clean = tr.strip()
            # 过滤掉错误信息和过短的内容
            if len(tr_clean) > 500 and 'error' not in tr_clean.lower()[:100]:
                # 截取合理长度用于 embedding
                text = tr_clean[:2000]
                samples.append(TextSample(
                    text=text,
                    category='raw_content',
                    source_file=file_name,
                    sample_id=sample_id,
                    char_count=len(tr_clean)
                ))

        # 提取 memory_summary（折叠后的记忆）
        summaries = re.findall(
            r'<memory_summary>(.*?)</memory_summary>',
            output,
            re.DOTALL
        )

        for ms in summaries:
            ms_clean = ms.strip()
            if len(ms_clean) > 100:
                samples.append(TextSample(
                    text=ms_clean[:2000],
                    category='folded_memory',
                    source_file=file_name,
                    sample_id=sample_id,
                    char_count=len(ms_clean)
                ))

    return samples


def extract_all_samples(gaia_dir: str) -> Tuple[List[TextSample], List[TextSample]]:
    """从所有结果文件中提取样本"""
    gaia_path = Path(gaia_dir)

    raw_samples = []
    folded_samples = []

    # 遍历所有 _evaluated.json 文件
    for f in sorted(gaia_path.glob('*_evaluated.json')):
        if 'overall' in f.name:
            continue

        print(f"Processing: {f.name}")
        samples = extract_samples_from_file(str(f))

        for s in samples:
            if s.category == 'raw_content':
                raw_samples.append(s)
            else:
                folded_samples.append(s)

    print(f"\nExtracted samples:")
    print(f"  Raw content: {len(raw_samples)}")
    print(f"  Folded memory: {len(folded_samples)}")

    return raw_samples, folded_samples


async def get_embeddings_batch(
    client: AsyncOpenAI,
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 20
) -> List[List[float]]:
    """批量获取文本 embeddings"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Getting embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        try:
            response = await client.embeddings.create(
                model=model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"  Error getting embeddings: {e}")
            # 使用零向量作为占位符
            all_embeddings.extend([[0.0] * 1536] * len(batch))

        await asyncio.sleep(0.1)  # Rate limiting

    return all_embeddings


def balance_samples(
    raw_samples: List[TextSample],
    folded_samples: List[TextSample],
    max_per_category: int = 100
) -> Tuple[List[TextSample], List[TextSample]]:
    """平衡样本数量，确保两类样本数量相近"""

    # 对 raw_samples 按字符数排序，选取最长的（更能代表网页噪声）
    raw_sorted = sorted(raw_samples, key=lambda x: x.char_count, reverse=True)

    # 去重（同一个 sample_id 只保留一个）
    seen_ids = set()
    raw_unique = []
    for s in raw_sorted:
        key = (s.source_file, s.sample_id)
        if key not in seen_ids:
            seen_ids.add(key)
            raw_unique.append(s)

    # 选取样本
    n_folded = min(len(folded_samples), max_per_category)
    n_raw = min(len(raw_unique), max_per_category, n_folded * 3)  # raw 可以多一些

    return raw_unique[:n_raw], folded_samples[:n_folded]


def compute_tsne(embeddings: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """计算 t-SNE 降维"""
    # 调整 perplexity 以适应样本数量
    n_samples = len(embeddings)
    adjusted_perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        n_iter=1000,
        random_state=42,
        learning_rate='auto',
        init='pca'
    )

    return tsne.fit_transform(embeddings)


def plot_tsne_visualization(
    raw_coords: np.ndarray,
    folded_coords: np.ndarray,
    output_path: str,
    title: str = "Memory Folding: Knowledge Representation Density"
):
    """绘制 t-SNE 可视化图"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制原始内容（蓝色，较大的点，表示分散）
    ax.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        c='#3498db',  # 蓝色
        s=60,
        alpha=0.6,
        label=f'Raw Web Content (n={len(raw_coords)})',
        edgecolors='white',
        linewidths=0.5
    )

    # 绘制折叠后的记忆（红色，较小的点，表示紧凑）
    ax.scatter(
        folded_coords[:, 0],
        folded_coords[:, 1],
        c='#e74c3c',  # 红色
        s=80,
        alpha=0.8,
        label=f'Folded Memory (n={len(folded_coords)})',
        edgecolors='white',
        linewidths=0.5,
        marker='s'  # 方形标记
    )

    # 添加图例
    ax.legend(
        loc='upper right',
        fontsize=11,
        framealpha=0.9,
        edgecolor='gray'
    )

    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)

    # 美化
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 添加说明文字
    textstr = 'Blue: Scattered raw web content\nRed: Compact knowledge representation'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_density_comparison(
    raw_coords: np.ndarray,
    folded_coords: np.ndarray,
    output_path: str
):
    """绘制密度对比图（双子图）"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：Raw Content
    ax1 = axes[0]
    ax1.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        c='#3498db',
        s=50,
        alpha=0.6,
        edgecolors='white',
        linewidths=0.3
    )
    ax1.set_title('(a) Raw Web Content', fontsize=13, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 计算分散度
    raw_std = np.std(raw_coords, axis=0).mean()
    ax1.text(0.05, 0.95, f'Spread: {raw_std:.2f}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 右图：Folded Memory
    ax2 = axes[1]
    ax2.scatter(
        folded_coords[:, 0],
        folded_coords[:, 1],
        c='#e74c3c',
        s=70,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.3,
        marker='s'
    )
    ax2.set_title('(b) Folded Memory', fontsize=13, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 计算分散度
    folded_std = np.std(folded_coords, axis=0).mean()
    ax2.text(0.05, 0.95, f'Spread: {folded_std:.2f}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # 调整两个子图使用相同的坐标范围
    all_coords = np.vstack([raw_coords, folded_coords])
    x_min, x_max = all_coords[:, 0].min() - 5, all_coords[:, 0].max() + 5
    y_min, y_max = all_coords[:, 1].min() - 5, all_coords[:, 1].max() + 5

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.suptitle('Memory Density Comparison via t-SNE', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compute_statistics(raw_samples: List[TextSample], folded_samples: List[TextSample]):
    """计算并打印统计信息"""
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)

    raw_chars = [s.char_count for s in raw_samples]
    folded_chars = [s.char_count for s in folded_samples]

    print(f"Raw Content:")
    print(f"  Count: {len(raw_samples)}")
    print(f"  Avg chars: {np.mean(raw_chars):.0f}")
    print(f"  Max chars: {np.max(raw_chars)}")
    print(f"  Min chars: {np.min(raw_chars)}")

    print(f"\nFolded Memory:")
    print(f"  Count: {len(folded_samples)}")
    print(f"  Avg chars: {np.mean(folded_chars):.0f}")
    print(f"  Max chars: {np.max(folded_chars)}")
    print(f"  Min chars: {np.min(folded_chars)}")

    compression_ratio = np.mean(raw_chars) / np.mean(folded_chars)
    print(f"\nCompression Ratio: {compression_ratio:.2f}x")
    print("="*60)


async def main():
    """主函数"""
    print("="*60)
    print("Memory Folding t-SNE Visualization")
    print("="*60)

    # 配置
    gaia_dir = "E:/websearcher/outputs/gaia"
    output_dir = "E:/websearcher/outputs/visualizations"
    config_path = "E:/websearcher/config/config.yaml"

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载配置
    config = load_config(config_path)

    # 初始化 OpenAI 客户端
    client = AsyncOpenAI(
        api_key=config['model']['auxiliary_model']['api_key'],
        base_url=config['model']['auxiliary_model']['api_base']
    )

    # 提取样本
    print("\n[1/5] Extracting samples...")
    raw_samples, folded_samples = extract_all_samples(gaia_dir)

    if len(folded_samples) == 0:
        print("No folded memory samples found!")
        return

    # 平衡样本
    print("\n[2/5] Balancing samples...")
    raw_samples, folded_samples = balance_samples(raw_samples, folded_samples, max_per_category=80)
    print(f"  After balancing: raw={len(raw_samples)}, folded={len(folded_samples)}")

    # 计算统计信息
    compute_statistics(raw_samples, folded_samples)

    # 获取 embeddings
    print("\n[3/5] Getting embeddings...")
    all_texts = [s.text for s in raw_samples] + [s.text for s in folded_samples]

    # 使用可用的 embedding 模型
    try:
        embeddings = await get_embeddings_batch(
            client,
            all_texts,
            model="text-embedding-3-small"  # 或使用其他可用模型
        )
    except Exception as e:
        print(f"Embedding API error: {e}")
        print("Trying alternative model...")
        embeddings = await get_embeddings_batch(
            client,
            all_texts,
            model="text-embedding-ada-002"
        )

    embeddings = np.array(embeddings)
    print(f"  Embedding shape: {embeddings.shape}")

    # 分割 embeddings
    n_raw = len(raw_samples)
    raw_embeddings = embeddings[:n_raw]
    folded_embeddings = embeddings[n_raw:]

    # t-SNE 降维
    print("\n[4/5] Computing t-SNE...")
    all_tsne = compute_tsne(embeddings, perplexity=min(30, len(all_texts) // 3))

    raw_tsne = all_tsne[:n_raw]
    folded_tsne = all_tsne[n_raw:]

    # 绘制可视化
    print("\n[5/5] Generating visualizations...")

    # 主图：合并视图
    plot_tsne_visualization(
        raw_tsne,
        folded_tsne,
        f"{output_dir}/memory_tsne_combined.png",
        title="Memory Folding: From Raw Web Content to Compact Knowledge"
    )

    # 对比图：双子图
    plot_density_comparison(
        raw_tsne,
        folded_tsne,
        f"{output_dir}/memory_tsne_comparison.png"
    )

    # 保存数据供后续分析
    np.savez(
        f"{output_dir}/tsne_data.npz",
        raw_tsne=raw_tsne,
        folded_tsne=folded_tsne,
        raw_embeddings=raw_embeddings,
        folded_embeddings=folded_embeddings
    )

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
