"""
WebSearcher 消融实验批量运行脚本
================================

使用方法:
    # 运行所有消融实验
    python run_ablation_batch.py --dataset data/GAIA/dev.json

    # 只运行指定的消融实验
    python run_ablation_batch.py --dataset data/GAIA/dev.json --configs no_working_memory no_episode_memory

    # 指定样本数
    python run_ablation_batch.py --dataset data/GAIA/dev.json --max_examples 50

    # 汇总已有结果
    python run_ablation_batch.py --summarize_only

可用的消融配置 (位于 config/ablation/):
    - no_episode_memory.yaml    : 移除 Episode Memory
    - no_working_memory.yaml    : 移除 Working Memory
    - no_tool_memory.yaml       : 移除 Tool Memory
    - no_memory_folding.yaml    : 禁用记忆折叠（直接截断）
    - no_intent_compression.yaml: 禁用意图引导压缩
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from evaluate import run_agent_on_dataset, evaluate_results


# 消融实验定义（不含baseline，主实验已跑）
ABLATION_EXPERIMENTS = {
    "no_hierarchical_memory": {
        "config": "config/ablation/no_hierarchical_memory.yaml",
        "description": "Without Hierarchical Memory"
    },
    "no_intent_interaction": {
        "config": "config/ablation/no_intent_interaction.yaml",
        "description": "Without Intent-Driven Interaction"
    },
    "single_model": {
        "config": "config/ablation/single_model.yaml",
        "description": "Without Dual-Model Orchestration"
    },
}


async def run_single_experiment(
    name: str,
    config_path: str,
    dataset_path: str,
    output_dir: str,
    max_examples: int,
    max_iterations: int
) -> dict:
    """运行单个消融实验"""

    print(f"\n{'='*70}")
    print(f"实验: {name}")
    print(f"配置: {config_path}")
    print(f"{'='*70}")

    # 检查配置文件是否存在
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return {"name": name, "status": "skipped", "error": "Config not found"}

    # 设置输出路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f"{name}_{timestamp}.json"

    try:
        # 运行Agent
        results_file = await run_agent_on_dataset(
            dataset_path=dataset_path,
            output_file=str(output_file),
            max_examples=max_examples,
            config_path=config_path,
            max_iterations=max_iterations,
            resume=True
        )

        # 评估结果
        await evaluate_results(
            results_file=results_file,
            config_path=config_path,
            output_suffix='_evaluated'
        )

        return {
            "name": name,
            "status": "completed",
            "results_file": results_file
        }

    except KeyboardInterrupt:
        print(f"\n⚠️ 实验 {name} 被中断")
        return {"name": name, "status": "interrupted"}

    except Exception as e:
        print(f"❌ 实验 {name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "status": "failed", "error": str(e)}


def summarize_results(output_dir: str) -> dict:
    """汇总所有消融实验结果"""

    results_dir = Path(output_dir)
    summary = {}

    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'实验名称':<25} {'成功率':>10} {'LLM准确率':>12}")
    print("-"*60)

    for exp_name in ABLATION_EXPERIMENTS.keys():
        # 查找该实验的评估结果
        pattern = f"{exp_name}_*_evaluated.overall.json"
        files = list(results_dir.glob(pattern))

        if not files:
            print(f"{exp_name:<25} {'--':>10} {'--':>12}")
            continue

        # 使用最新文件
        latest = max(files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            # 加载详细结果
            detail_file = str(latest).replace('.overall.json', '.json')
            success_rate = 0

            if Path(detail_file).exists():
                with open(detail_file, 'r', encoding='utf-8') as f:
                    details = json.load(f)

                total = len(details)
                if total > 0:
                    success_rate = sum(1 for d in details if d.get('Success')) / total * 100

            llm_acc = metrics.get('llm_equal', 0) * 100

            summary[exp_name] = {
                "success_rate": success_rate,
                "llm_accuracy": llm_acc,
                "metrics": metrics
            }

            print(f"{exp_name:<25} {success_rate:>9.1f}% {llm_acc:>11.1f}%")

        except Exception as e:
            print(f"{exp_name:<25} Error: {e}")

    print("="*60)

    # 保存汇总
    summary_file = results_dir / "ablation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_file}")

    # 生成LaTeX表格
    generate_latex_table(summary, results_dir / "ablation_table.tex")

    return summary


def generate_latex_table(summary: dict, output_path: Path):
    """生成LaTeX表格"""

    latex = r"""\begin{table}[t]
\centering
\caption{Ablation study on GAIA Level 3.}
\label{tab:ablation}
\begin{tabular}{lc}
\toprule
\textbf{Configuration} & \textbf{Success (\%)} \\
\midrule
"""

    # Baseline first
    if 'baseline' in summary:
        s = summary['baseline']
        latex += f"\\textbf{{Full Model}} & \\textbf{{{s['success_rate']:.1f}}} \\\\\n"
        latex += "\\midrule\n"

    # Ablations
    for name in ['no_hierarchical_memory', 'no_intent_interaction', 'single_model']:
        if name in summary:
            s = summary[name]
            desc = ABLATION_EXPERIMENTS[name]['description'].replace('Without ', 'w/o ')
            latex += f"{desc} & {s['success_rate']:.1f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"LaTeX表格已保存: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="WebSearcher 消融实验批量运行",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset', type=str, default='data/GAIA/dev.json',
                       help='数据集路径')
    parser.add_argument('--max_examples', type=int, default=-1,
                       help='最大样本数 (-1=全部)')
    parser.add_argument('--max_iterations', type=int, default=20,
                       help='最大迭代次数')
    parser.add_argument('--output_dir', type=str, default='outputs/ablation',
                       help='输出目录')
    parser.add_argument('--configs', type=str, nargs='+',
                       choices=list(ABLATION_EXPERIMENTS.keys()),
                       help='要运行的消融实验')
    parser.add_argument('--summarize_only', action='store_true',
                       help='只汇总已有结果')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize_results(args.output_dir)
        return

    # 确定要运行的实验
    experiments = args.configs if args.configs else list(ABLATION_EXPERIMENTS.keys())

    print(f"\n{'#'*70}")
    print(f"# WebSearcher 消融实验")
    print(f"# 数据集: {args.dataset}")
    print(f"# 样本数: {args.max_examples if args.max_examples > 0 else '全部'}")
    print(f"# 实验: {', '.join(experiments)}")
    print(f"{'#'*70}\n")

    # 运行实验
    results = []
    for i, exp_name in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] 运行: {exp_name}")

        exp_info = ABLATION_EXPERIMENTS[exp_name]
        result = await run_single_experiment(
            name=exp_name,
            config_path=exp_info['config'],
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_examples=args.max_examples,
            max_iterations=args.max_iterations
        )
        results.append(result)

        # 检查是否中断
        if result['status'] == 'interrupted':
            print("\n用户中断，停止后续实验")
            break

    # 汇总结果
    print("\n\n" + "#"*70)
    print("# 所有实验完成，汇总结果")
    print("#"*70)

    summarize_results(args.output_dir)


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
