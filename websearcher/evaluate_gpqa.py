

















import json
import asyncio
import yaml
import sys
import argparse
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict
from typing import List, Tuple, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

                 
sys.path.append(str(Path(__file__).parent))

from src.run_agent import run_single_question
from src.evaluate.metrics import extract_answer_fn


_CHOICE_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_CHOICE_FALLBACK_RE = re.compile(r"([A-D])(?=[\s\)\]\}\.\,:;]|$)", re.IGNORECASE)


def _extract_choice_letter(text: str) -> str:
    """
    Extract a GPQA multiple-choice letter (A/B/C/D) from a free-form string.
    Returns '' if no valid choice is found.
    """
    if not text:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # Light LaTeX / wrapper cleanup
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)

    matches = _CHOICE_RE.findall(s)
    if matches:
        return matches[-1].upper()

    matches = _CHOICE_FALLBACK_RE.findall(s)
    if matches:
        return matches[-1].upper()

    return ""


def _is_llm_correct(response_text: str) -> bool:
    """
    Parse LLM response. Expect "Correct"/"Incorrect" (case-insensitive).
    Be defensive: treat any mention of incorrect/wrong as Incorrect.
    """
    if not response_text:
        return False
    t = response_text.strip().lower()
    if "incorrect" in t or "wrong" in t or "not correct" in t:
        return False
    return "correct" in t


async def _llm_eval_gpqa_single(
    client: AsyncOpenAI,
    question: str,
    correct_choice: str,
    correct_answer: str,
    agent_output: str,
    pred_choice: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
) -> Tuple[bool, str]:
    # Trim very long output to keep evaluation fast/cheap while still showing final reasoning/answer.
    if agent_output and len(agent_output) > 6000:
        agent_output = "...(前面的推理过程省略)...\n\n" + agent_output[-6000:]

    correct_choice = _extract_choice_letter(correct_choice)
    pred_choice = _extract_choice_letter(pred_choice)

    prompt = f"""你是一个专业的答案评估助手。请判断 Agent 在 GPQA 多选题上的回答是否正确。

# 题目（包含选项）
{question}

# 标准答案
- 正确选项字母: {correct_choice}
- 正确答案内容: {correct_answer}

# Agent 输出（推理过程/原文）
{agent_output}

# Agent 提取到的答案（可能是字母或文本）
- Pred_Choice: {pred_choice}
- Pred_Answer: {pred_answer}

# 你的任务
请判断 Agent 最终选择是否与标准答案一致。
- 如果 Agent 明确选择了正确选项字母（A/B/C/D），判为 Correct。
- 如果 Agent 没有给出字母，但其最终结论与“正确答案内容”一致，也判为 Correct。
- 只要不一致或无法判断为一致，就判为 Incorrect。

# 回答格式
只回答 "Correct" 或 "Incorrect"，不要有任何解释。"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
            text = (resp.choices[0].message.content or "").strip()
            return _is_llm_correct(text), text
        except Exception as e:
            if attempt == retry_limit - 1:
                return False, f"Error: {e}"
            await asyncio.sleep(2 * (attempt + 1))

    return False, "Error"


async def _llm_eval_gpqa_batch(
    questions: List[str],
    correct_choices: List[str],
    correct_answers: List[str],
    outputs: List[str],
    pred_choices: List[str],
    pred_answers: List[str],
    api_base: str,
    model_name: str,
    api_key: str,
    concurrent_limit: int = 5,
) -> List[Tuple[bool, str]]:
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    semaphore = asyncio.Semaphore(concurrent_limit)

    async def run_one(i: int):
        return i, await _llm_eval_gpqa_single(
            client=client,
            question=questions[i],
            correct_choice=correct_choices[i],
            correct_answer=correct_answers[i],
            agent_output=outputs[i],
            pred_choice=pred_choices[i],
            pred_answer=pred_answers[i],
            model_name=model_name,
            semaphore=semaphore,
        )

    tasks = [run_one(i) for i in range(len(questions))]
    indexed_results: List[Tuple[int, Tuple[bool, str]]] = []
    with tqdm(total=len(tasks), desc="LLM评估(GPQA)", ncols=100) as pbar:
        for coro in asyncio.as_completed(tasks):
            i, res = await coro
            indexed_results.append((i, res))
            pbar.update(1)

    indexed_results.sort(key=lambda x: x[0])
    results = [r for _, r in indexed_results]
    await client.close()
    return results


                                                           

async def run_agent_on_gpqa(
    dataset_path: str = 'data/GPQA/diamond.json',
    output_file: str = None,
    max_examples: int = -1,
    start_idx: int = 0,
    config_path: str = 'config/config.yaml',
    max_iterations: int = 30,
    resume: bool = True
):

    
    print("="*80)
    print("运行Agent - GPQA Dataset")
    print("="*80)
    
           
    print(f"\n加载数据集: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
           
    if max_examples > 0:
        dataset = dataset[start_idx:start_idx + max_examples]
    else:
        dataset = dataset[start_idx:]
    
    total = len(dataset)
    print(f"处理样本数: {total}")
    print(f"起始索引: {start_idx}")
    
            
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'outputs/gpqa/gpqa_results_{timestamp}.json'
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
                      
    results = []
    processed_ids = set()
    resume_from_idx = 0
    
    if resume and output_path.exists():
        print(f"\n检测到已有结果文件: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            for item in results:
                processed_ids.add(item.get('id'))
            
            resume_from_idx = len(results)
            print(f"已完成 {resume_from_idx} 个样本，将从第 {resume_from_idx + 1} 个继续")
            
            if resume_from_idx >= total:
                print(f"所有样本已处理完成！")
                return output_file
                
        except Exception as e:
            print(f"读取已有结果失败: {e}")
            print("将从头开始处理")
            results = []
            processed_ids = set()
            resume_from_idx = 0
    
    success_count = sum(1 for r in results if r.get('Success', False))
    
    print(f"\n{'='*80}")
    print("开始处理")
    print(f"{'='*80}\n")
    
             
    for idx in range(resume_from_idx, total):
        example = dataset[idx]
        example_id = example.get('id', start_idx + idx)
        
        if example_id in processed_ids:
            print(f"跳过已处理的样本 {example_id}")
            continue
        
        question = example.get('Question', '')
        correct_answer = example.get('Correct Answer', '')
        correct_choice = example.get('Correct Choice', '')
        
        print(f"\n[{idx + 1}/{total}] 处理样本 {example_id}")
        print(f"问题: {question[:100]}...")
        print(f"正确答案: {correct_answer} ({correct_choice})")
        
        try:
                     
            result = await run_single_question(
                question=question,
                config_path=config_path,
                max_iterations=max_iterations,
                max_context_tokens=30000
            )
            
                  
            # GPQA is multiple-choice; extract as choose to better handle \text{A} etc.
            pred_answer = extract_answer_fn(result['answer'], mode='choose')
            
                  
            result_item = {
                'id': example_id,
                'Question': question,
                'Correct_Answer': correct_answer,
                'Correct_Choice': correct_choice,
                'Subdomain': example.get('Subdomain', ''),
                'High_level_domain': example.get('High-level domain', ''),
                'Output': result['reasoning'],
                'Pred_Answer': pred_answer,
                'Success': result['success'],
                'Iterations': result['iterations'],
                'Tool_Calls': result['tool_calls'],
                'Memory_Folds': result['memory_folds'],
                'Total_Tokens': result['total_tokens'],
                'WebExplorer': []
            }
            
                    
            for interaction in result.get('interactions', []):
                if interaction['type'] == 'tool_call':
                    result_item['WebExplorer'].append({
                        'tool_name': interaction['tool_name'],
                        'Input': interaction['arguments'],
                        'Output': interaction['result'][:200] if isinstance(interaction['result'], str) else str(interaction['result'])[:200],
                        'iteration': interaction['iteration']
                    })
            
            results.append(result_item)
            
            if result['success']:
                success_count += 1
            
            print(f"✓ 完成 - 成功: {result['success']}, 迭代: {result['iterations']}, 工具调用: {result['tool_calls']}")
            
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("检测到中断信号 (Ctrl+C)")
            print(f"{'='*80}")
            print(f"已完成: {len(results)}/{total} 个样本")
            print(f"成功: {success_count}")
            print(f"结果已保存至: {output_file}")
            print(f"\n要继续评估，请再次运行相同的命令")
            print(f"程序会自动从第 {len(results) + 1} 个样本继续")
            print(f"{'='*80}\n")
            return output_file
            
        except Exception as e:
            print(f"\n✗ 处理样本 {example_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            
            result_item = {
                'id': example_id,
                'Question': question,
                'Correct_Answer': correct_answer,
                'Correct_Choice': correct_choice,
                'Subdomain': example.get('Subdomain', ''),
                'High_level_domain': example.get('High-level domain', ''),
                'Output': '',
                'Pred_Answer': '',
                'Success': False,
                'Error': str(e),
                'Iterations': 0,
                'Tool_Calls': 0,
                'Memory_Folds': 0,
                'Total_Tokens': 0,
                'WebExplorer': []
            }
            results.append(result_item)
        
                    
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
              
        if (len(results) % 5 == 0):
            print(f"\n--- 进度: {len(results)}/{total} ({len(results)/total:.1%}) ---")
            print(f"成功: {success_count}/{len(results)} ({success_count/len(results):.1%})")
            print(f"结果已保存\n")
    
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Agent执行完成")
    print(f"{'='*80}")
    print(f"总样本数: {total}")
    print(f"成功: {success_count} ({success_count/total:.1%})")
    print(f"失败: {total - success_count} ({(total - success_count)/total:.1%})")
    print(f"结果保存至: {output_file}")
    print(f"{'='*80}\n")
    
    return output_file


                                                      

def evaluate_gpqa_results(
    results_file: str,
    config_path: str = 'config/config.yaml',
    output_suffix: str = '_evaluated'
):

    
    print("="*80)
    print("评估GPQA结果")
    print("="*80)
    
    print("\n加载配置...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    eval_model = config.get('model', {}).get('evaluation_model', {})
    if not eval_model:
        eval_model = config.get('model', {}).get('auxiliary_model', {})

    api_base = eval_model.get('api_base', 'https://api.openai.com/v1')
    model_name = eval_model.get('name', 'deepseek-v3.2')
    api_key = eval_model.get('api_key', 'empty')
    concurrent_limit = int(eval_model.get('concurrent_limit', 5))

    print(f"评估模型: {model_name}")

    print(f"\n加载结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    results_path = Path(results_file)
    base_name = results_path.stem
    output_dir = results_path.parent

    output_metrics_path = f"{base_name}{output_suffix}.json"
    output_metrics_overall_path = f"{base_name}{output_suffix}.overall.json"

    domain_fields = [
        'High-level domain',
        'High_level_domain',
        'Subdomain',
        'subdomain',
        'Level',
        'level',
        'category',
        'difficulty_level',
        'field',
        'problem_topic'
    ]

    if isinstance(results, dict):
        filtered_data = [v for v in results.values() if isinstance(v, dict)]
    else:
        filtered_data = results

    def get_domain(item: dict) -> str:
        for field in domain_fields:
            if field in item and item[field] is not None:
                return str(item[field])
        return "Unknown"

    # First pass: deterministic multiple-choice extraction (辅助字段，便于诊断)
    correct_flags = []
    valid_flags = []
    domain_bucket = defaultdict(list)  # domain -> list[int correct]

    for item in filtered_data:
        gt_choice_raw = (
            item.get("Correct_Choice")
            or item.get("Correct Choice")
            or item.get("answer_letter")
            or item.get("answer")
            or ""
        )
        gt_choice = _extract_choice_letter(gt_choice_raw)

        pred_choice = _extract_choice_letter(item.get("Pred_Answer", ""))
        if not pred_choice:
            # Fallback: try extracting from full output
            output_text = item.get("Output", "") or item.get("result", "") or ""
            extracted = extract_answer_fn(output_text, mode="choose") or ""
            pred_choice = _extract_choice_letter(extracted) or _extract_choice_letter(output_text[-500:])

        is_valid = int(bool(pred_choice))
        is_correct = int(bool(pred_choice) and bool(gt_choice) and pred_choice == gt_choice)

        item["Pred_Choice"] = pred_choice
        item["GT_Choice"] = gt_choice
        item.setdefault("Metrics", {})
        item["Metrics"].update(
            {
                "is_valid_answer": is_valid,
                # keep both names for compatibility with other evaluators/plots
                "choice_acc": is_correct,
                "acc": is_correct,
            }
        )

        correct_flags.append(is_correct)
        valid_flags.append(is_valid)
        domain_bucket[get_domain(item)].append(is_correct)

    total = len(filtered_data)
    overall_metrics = {
        "total": total,
        # keep both names for compatibility with other evaluators/plots
        "choice_acc": (sum(correct_flags) / total) if total > 0 else 0.0,
        "acc": (sum(correct_flags) / total) if total > 0 else 0.0,
        "num_valid_answer": f"{sum(valid_flags)} of {total}",
        "domain_metrics": {
            domain: {
                "total": len(vals),
                "choice_acc": (sum(vals) / len(vals)) if vals else 0.0,
                "acc": (sum(vals) / len(vals)) if vals else 0.0,
            }
            for domain, vals in domain_bucket.items()
        },
    }

    # Second pass (重点): always run LLM evaluation
    print("\n运行LLM评估（重点指标 llm_equal）...")
    questions: List[str] = []
    correct_choices: List[str] = []
    correct_answers: List[str] = []
    outputs: List[str] = []
    pred_choices: List[str] = []
    pred_answers: List[str] = []
    valid_indices: List[int] = []

    for i, item in enumerate(filtered_data):
        q = item.get("Question", "") or item.get("question", "") or ""
        cc = item.get("Correct_Choice") or item.get("Correct Choice") or ""
        ca = item.get("Correct_Answer") or item.get("Correct Answer") or ""
        out = item.get("Output", "") or item.get("result", "") or ""
        pa = item.get("Pred_Answer", "") or ""
        pc = item.get("Pred_Choice", "") or _extract_choice_letter(pa)

        # We require at least question + something labeled + some model output.
        if q and (cc or ca) and out:
            questions.append(q)
            correct_choices.append(str(cc))
            correct_answers.append(str(ca))
            outputs.append(str(out))
            pred_choices.append(str(pc))
            pred_answers.append(str(pa))
            valid_indices.append(i)
        else:
            item.setdefault("Metrics", {})
            item["Metrics"].update(
                {
                    "llm_equal": 0,
                    "llm_response": "Skipped: missing question/label/output",
                }
            )

    llm_equal_flags: List[int] = [0] * total
    if valid_indices:
        llm_results = asyncio.run(
            _llm_eval_gpqa_batch(
                questions=questions,
                correct_choices=correct_choices,
                correct_answers=correct_answers,
                outputs=outputs,
                pred_choices=pred_choices,
                pred_answers=pred_answers,
                api_base=api_base,
                model_name=model_name,
                api_key=api_key,
                concurrent_limit=concurrent_limit,
            )
        )

        for idx_in_list, (is_correct, resp_text) in enumerate(llm_results):
            i = valid_indices[idx_in_list]
            item = filtered_data[i]
            item.setdefault("Metrics", {})
            item["Metrics"]["llm_equal"] = int(bool(is_correct))
            item["Metrics"]["llm_response"] = resp_text
            llm_equal_flags[i] = int(bool(is_correct))

    overall_metrics["num_llm_evaluated"] = f"{len(valid_indices)} of {total}"
    overall_metrics["llm_equal"] = (sum(llm_equal_flags) / total) if total > 0 else 0.0

    # LLM by domain
    llm_domain_bucket = defaultdict(list)
    for item in filtered_data:
        llm_domain_bucket[get_domain(item)].append(int(item.get("Metrics", {}).get("llm_equal", 0)))
    overall_metrics["llm_domain_metrics"] = {
        domain: {
            "total": len(vals),
            "llm_equal": (sum(vals) / len(vals)) if vals else 0.0,
        }
        for domain, vals in llm_domain_bucket.items()
    }

    # Save outputs
    with open(output_dir / output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    with open(output_dir / output_metrics_overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)

    detailed_file = output_dir / output_metrics_path
    overall_file = output_dir / output_metrics_overall_path

    print("\n" + "="*80)
    print("评估完成!")
    print(f"total: {overall_metrics['total']}")
    print(f"llm_equal: {overall_metrics.get('llm_equal', 0.0):.2%}" if overall_metrics["total"] else "llm_equal: 0.00%")
    print(f"choice_acc: {overall_metrics['choice_acc']:.2%}" if overall_metrics["total"] else "choice_acc: 0.00%")
    print("="*80)

    return detailed_file, overall_file


                                                

def main():
    parser = argparse.ArgumentParser(
        description="GPQA数据集评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--eval_only', action='store_true', help='只评估已有结果')
    mode_group.add_argument('--run_only', action='store_true', help='只运行agent，不评估')
    
    parser.add_argument('--dataset', type=str, default='data/GPQA/diamond.json', help='数据集路径')
    parser.add_argument('--max_examples', type=int, default=-1, help='最大处理样本数')
    parser.add_argument('--start_idx', type=int, default=0, help='起始索引')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--max_iterations', type=int, default=30, help='最大迭代次数')
    parser.add_argument('--output_dir', type=str, default='outputs/gpqa', help='输出目录')
    parser.add_argument('--output_name', type=str, default=None, help='输出文件名')
    parser.add_argument('--results', type=str, help='要评估的结果文件')
    parser.add_argument('--output_suffix', type=str, default='_evaluated', help='评估输出文件后缀')
    
    args = parser.parse_args()
    
    if args.eval_only and not args.results:
        parser.error("--eval_only 需要 --results 参数")
    
    if args.output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_name = f'gpqa_results_{timestamp}.json'
    
    output_file = Path(args.output_dir) / args.output_name
    
    print("\n" + "="*80)
    print("GPQA 数据集评估脚本")
    print("="*80)
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    if args.eval_only:
        evaluate_gpqa_results(
            results_file=args.results,
            config_path=args.config,
            output_suffix=args.output_suffix
        )
    elif args.run_only:
        asyncio.run(run_agent_on_gpqa(
            dataset_path=args.dataset,
            output_file=str(output_file),
            max_examples=args.max_examples,
            start_idx=args.start_idx,
            config_path=args.config,
            max_iterations=args.max_iterations
        ))
    else:
        results_file = asyncio.run(run_agent_on_gpqa(
            dataset_path=args.dataset,
            output_file=str(output_file),
            max_examples=args.max_examples,
            start_idx=args.start_idx,
            config_path=args.config,
            max_iterations=args.max_iterations
        ))
        
        evaluate_gpqa_results(
            results_file=results_file,
            config_path=args.config,
            output_suffix=args.output_suffix
        )


if __name__ == "__main__":
    main()
