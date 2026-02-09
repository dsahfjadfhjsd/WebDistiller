"""
WebSearcher 消融实验运行脚本
============================

运行方式:
    # 运行所有消融实验
    python run_ablation.py --dataset data/GAIA/dev.json --max_examples 50

    # 运行指定消融实验
    python run_ablation.py --dataset data/GAIA/dev.json --ablation no_episode_memory

    # 只评估已有结果
    python run_ablation.py --eval_only --output_dir outputs/ablation

消融实验列表:
    1. full_model              - 完整模型（baseline）
    2. no_episode_memory       - 移除 Episode Memory
    3. no_working_memory       - 移除 Working Memory
    4. no_tool_memory          - 移除 Tool Memory
    5. no_memory_folding       - 禁用记忆折叠（直接截断）
    6. no_intent_extraction    - 禁用意图提取
    7. no_intent_compression   - 禁用意图引导压缩
    8. single_model            - 单模型（不使用辅助模型）
    9. no_parallel_retrieval   - 禁用并行批量检索
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy

sys.path.append(str(Path(__file__).parent))

from evaluate import run_agent_on_dataset, evaluate_results


# ============================================================================
# 消融实验配置定义
# ============================================================================

ABLATION_CONFIGS = {
    # Baseline - 完整模型
    "full_model": {
        "description": "Full model (baseline)",
        "changes": {}  # 无修改
    },

    # ==================== 记忆组件消融 ====================
    "no_episode_memory": {
        "description": "Without Episode Memory",
        "changes": {
            "memory.episode_memory.enabled": False
        }
    },

    "no_working_memory": {
        "description": "Without Working Memory",
        "changes": {
            "memory.working_memory.enabled": False
        }
    },

    "no_tool_memory": {
        "description": "Without Tool Memory",
        "changes": {
            "memory.tool_memory.enabled": False
        }
    },

    "no_memory_folding": {
        "description": "Without Memory Folding (truncation only)",
        "changes": {
            "memory.fold_threshold": 0.0,  # 永不触发折叠
            "memory.episode_memory.enabled": False,
            "memory.working_memory.enabled": False,
            "memory.tool_memory.enabled": False
        }
    },

    # ==================== Intent机制消融 ====================
    "no_intent_extraction": {
        "description": "Without Intent Extraction",
        "changes": {
            "ablation.disable_intent_extraction": True
        }
    },

    "no_intent_compression": {
        "description": "Without Intent-Guided Compression",
        "changes": {
            "memory.auto_summarize_tool_results.enabled": False,
            "ablation.disable_intent_compression": True
        }
    },

    "no_intent_in_memory": {
        "description": "Without Intent in Memory Folding",
        "changes": {
            "ablation.disable_intent_in_memory": True
        }
    },

    # ==================== 架构消融 ====================
    "single_model": {
        "description": "Single Model (no auxiliary model)",
        "changes": {
            "ablation.single_model": True,
            # 辅助模型设置为与主模型相同
            "model.auxiliary_model.name": "${model.main_model.name}",
            "model.auxiliary_model.api_base": "${model.main_model.api_base}",
            "model.auxiliary_model.api_key": "${model.main_model.api_key}",
            "model.auxiliary_model.temperature": "${model.main_model.temperature}"
        }
    },

    "no_parallel_retrieval": {
        "description": "Without Parallel Batch Retrieval (sequential)",
        "changes": {
            "ablation.max_concurrent_fetch": 1
        }
    },
}


def set_nested_value(d: dict, key_path: str, value):
    """设置嵌套字典的值，支持点号分隔的路径"""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_value(d: dict, key_path: str, default=None):
    """获取嵌套字典的值"""
    keys = key_path.split('.')
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def resolve_references(config: dict, value):
    """解析配置中的引用（如 ${model.main_model.name}）"""
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        ref_path = value[2:-1]
        return get_nested_value(config, ref_path)
    return value


def create_ablation_config(base_config: dict, ablation_name: str) -> dict:
    """基于基础配置创建消融实验配置"""
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    ablation = ABLATION_CONFIGS[ablation_name]
    config = deepcopy(base_config)

    # 确保ablation部分存在
    if 'ablation' not in config:
        config['ablation'] = {}

    # 应用变更
    for key_path, value in ablation['changes'].items():
        resolved_value = resolve_references(config, value)
        set_nested_value(config, key_path, resolved_value)

    # 添加元信息
    config['ablation']['name'] = ablation_name
    config['ablation']['description'] = ablation['description']

    return config


def save_ablation_config(config: dict, output_path: str):
    """保存消融配置到yaml文件"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return output_path


# ============================================================================
# 消融实验运行
# ============================================================================

async def run_single_ablation(
    ablation_name: str,
    dataset_path: str,
    base_config_path: str,
    output_dir: str,
    max_examples: int = -1,
    max_iterations: int = 20
):
    """运行单个消融实验"""
    print(f"\n{'='*80}")
    print(f"消融实验: {ablation_name}")
    print(f"描述: {ABLATION_CONFIGS[ablation_name]['description']}")
    print(f"{'='*80}")

    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # 创建消融配置
    ablation_config = create_ablation_config(base_config, ablation_name)

    # 保存消融配置
    config_dir = Path(output_dir) / 'configs'
    config_path = config_dir / f'{ablation_name}.yaml'
    save_ablation_config(ablation_config, str(config_path))
    print(f"配置已保存: {config_path}")

    # 设置输出文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / 'results' / f'{ablation_name}_{timestamp}.json'

    # 运行实验
    try:
        results_file = await run_agent_on_dataset(
            dataset_path=dataset_path,
            output_file=str(output_file),
            max_examples=max_examples,
            config_path=str(config_path),
            max_iterations=max_iterations,
            resume=True
        )

        # 评估结果
        await evaluate_results(
            results_file=results_file,
            config_path=str(config_path),
            output_suffix='_evaluated'
        )

        return results_file

    except Exception as e:
        print(f"实验 {ablation_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_all_ablations(
    dataset_path: str,
    base_config_path: str,
    output_dir: str,
    max_examples: int = -1,
    max_iterations: int = 20,
    ablations: list = None
):
    """运行所有或指定的消融实验"""

    if ablations is None:
        ablations = list(ABLATION_CONFIGS.keys())

    print(f"\n{'#'*80}")
    print(f"# WebSearcher 消融实验")
    print(f"# 数据集: {dataset_path}")
    print(f"# 样本数: {max_examples if max_examples > 0 else '全部'}")
    print(f"# 实验数: {len(ablations)}")
    print(f"{'#'*80}\n")

    results = {}

    for i, ablation_name in enumerate(ablations):
        print(f"\n[{i+1}/{len(ablations)}] 运行实验: {ablation_name}")

        result_file = await run_single_ablation(
            ablation_name=ablation_name,
            dataset_path=dataset_path,
            base_config_path=base_config_path,
            output_dir=output_dir,
            max_examples=max_examples,
            max_iterations=max_iterations
        )

        results[ablation_name] = result_file

    # 汇总结果
    print(f"\n{'#'*80}")
    print("# 消融实验完成")
    print(f"{'#'*80}")

    summary = aggregate_ablation_results(output_dir, ablations)

    return results, summary


def aggregate_ablation_results(output_dir: str, ablations: list) -> dict:
    """汇总所有消融实验结果"""
    results_dir = Path(output_dir) / 'results'
    summary = {}

    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    print(f"{'实验名称':<25} {'Success%':>10} {'Avg Steps':>10} {'Tokens(k)':>12} {'LLM Equal%':>12}")
    print("-"*80)

    for ablation_name in ablations:
        # 查找该实验的评估结果文件
        pattern = f"{ablation_name}_*_evaluated.overall.json"
        matching_files = list(results_dir.glob(pattern))

        if not matching_files:
            print(f"{ablation_name:<25} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
            continue

        # 使用最新的结果
        latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            # 同时加载详细结果计算额外指标
            detail_file = str(latest_file).replace('.overall.json', '.json')
            avg_steps = 0
            avg_tokens = 0
            success_rate = 0

            if Path(detail_file).exists():
                with open(detail_file, 'r', encoding='utf-8') as f:
                    details = json.load(f)

                total = len(details)
                if total > 0:
                    avg_steps = sum(d.get('Iterations', 0) for d in details) / total
                    avg_tokens = sum(d.get('Total_Tokens', 0) for d in details) / total / 1000
                    success_rate = sum(1 for d in details if d.get('Success', False)) / total * 100

            llm_equal = metrics.get('llm_equal', 0) * 100

            summary[ablation_name] = {
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_tokens_k': avg_tokens,
                'llm_equal': llm_equal,
                'metrics': metrics
            }

            print(f"{ablation_name:<25} {success_rate:>9.1f}% {avg_steps:>10.1f} {avg_tokens:>11.1f}k {llm_equal:>11.1f}%")

        except Exception as e:
            print(f"{ablation_name:<25} Error: {e}")

    print("="*80)

    # 保存汇总结果
    summary_file = Path(output_dir) / 'ablation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存: {summary_file}")

    return summary


def generate_latex_table(summary: dict, output_path: str = None):
    """生成LaTeX格式的消融实验表格"""

    latex = r"""
\begin{table}[t]
\centering
\caption{Ablation study on GAIA Level 3. We report success rate, average reasoning steps, and total token consumption.}
\label{tab:ablation}
\begin{adjustbox}{width=\columnwidth}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Success} & \textbf{Avg. Steps} & \textbf{Tokens (k)} \\
\midrule
"""

    # Full model first
    if 'full_model' in summary:
        s = summary['full_model']
        latex += f"\\textbf{{Full Model}} & \\textbf{{{s['success_rate']:.1f}\\%}} & {s['avg_steps']:.1f} & {s['avg_tokens_k']:.1f} \\\\\n"
        latex += "\\midrule\n"

    # Memory components
    latex += "\\multicolumn{4}{l}{\\textit{Memory Components}} \\\\\n"
    for name in ['no_episode_memory', 'no_working_memory', 'no_tool_memory', 'no_memory_folding']:
        if name in summary:
            s = summary[name]
            display_name = ABLATION_CONFIGS[name]['description'].replace('Without ', 'w/o ')
            latex += f"\\quad {display_name} & {s['success_rate']:.1f}\\% & {s['avg_steps']:.1f} & {s['avg_tokens_k']:.1f} \\\\\n"

    # Intent mechanism
    latex += "\\midrule\n"
    latex += "\\multicolumn{4}{l}{\\textit{Intent Mechanism}} \\\\\n"
    for name in ['no_intent_extraction', 'no_intent_compression', 'no_intent_in_memory']:
        if name in summary:
            s = summary[name]
            display_name = ABLATION_CONFIGS[name]['description'].replace('Without ', 'w/o ')
            latex += f"\\quad {display_name} & {s['success_rate']:.1f}\\% & {s['avg_steps']:.1f} & {s['avg_tokens_k']:.1f} \\\\\n"

    # Architecture
    latex += "\\midrule\n"
    latex += "\\multicolumn{4}{l}{\\textit{Architecture}} \\\\\n"
    for name in ['single_model', 'no_parallel_retrieval']:
        if name in summary:
            s = summary[name]
            display_name = ABLATION_CONFIGS[name]['description']
            latex += f"\\quad {display_name} & {s['success_rate']:.1f}\\% & {s['avg_steps']:.1f} & {s['avg_tokens_k']:.1f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"LaTeX表格已保存: {output_path}")

    return latex


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WebSearcher 消融实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验列表:
  full_model            - 完整模型（baseline）
  no_episode_memory     - 移除 Episode Memory
  no_working_memory     - 移除 Working Memory
  no_tool_memory        - 移除 Tool Memory
  no_memory_folding     - 禁用记忆折叠
  no_intent_extraction  - 禁用意图提取
  no_intent_compression - 禁用意图引导压缩
  single_model          - 单模型
  no_parallel_retrieval - 禁用并行检索

使用示例:
  # 运行所有消融实验
  python run_ablation.py --dataset data/GAIA/dev.json --max_examples 50

  # 运行指定实验
  python run_ablation.py --dataset data/GAIA/dev.json --ablation no_working_memory

  # 只汇总已有结果
  python run_ablation.py --summarize_only --output_dir outputs/ablation
        """
    )

    # 运行模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--summarize_only', action='store_true',
                           help='只汇总已有结果')
    mode_group.add_argument('--eval_only', action='store_true',
                           help='只评估已有结果')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='data/GAIA/dev.json',
                       help='数据集路径')
    parser.add_argument('--max_examples', type=int, default=-1,
                       help='最大样本数（-1表示全部）')

    # 配置参数
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='基础配置文件路径')
    parser.add_argument('--max_iterations', type=int, default=20,
                       help='最大迭代次数')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs/ablation',
                       help='输出目录')

    # 消融选择
    parser.add_argument('--ablation', type=str, nargs='+',
                       choices=list(ABLATION_CONFIGS.keys()),
                       help='指定要运行的消融实验')

    # LaTeX输出
    parser.add_argument('--latex', action='store_true',
                       help='生成LaTeX表格')

    args = parser.parse_args()

    # Windows事件循环
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 确定要运行的消融实验
    ablations = args.ablation if args.ablation else list(ABLATION_CONFIGS.keys())

    if args.summarize_only:
        # 只汇总结果
        summary = aggregate_ablation_results(args.output_dir, ablations)
        if args.latex:
            latex_path = Path(args.output_dir) / 'ablation_table.tex'
            generate_latex_table(summary, str(latex_path))
    else:
        # 运行实验
        results, summary = asyncio.run(run_all_ablations(
            dataset_path=args.dataset,
            base_config_path=args.config,
            output_dir=args.output_dir,
            max_examples=args.max_examples,
            max_iterations=args.max_iterations,
            ablations=ablations
        ))

        if args.latex:
            latex_path = Path(args.output_dir) / 'ablation_table.tex'
            generate_latex_table(summary, str(latex_path))


if __name__ == "__main__":
    main()
