

















import json
import asyncio
import yaml
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import AsyncOpenAI

                 
sys.path.append(str(Path(__file__).parent))

from src.run_agent import run_single_question
from src.evaluate.metrics import extract_answer_fn, normalize_answer_qa
from src.utils.math_equivalence import is_equiv


                                                           

async def run_agent_on_webwalker(
    dataset_path: str = 'data/WebWalkerQA/test.json',
    output_file: str = None,
    max_examples: int = -1,
    start_idx: int = 0,
    config_path: str = 'config/config.yaml',
    max_iterations: int = 25,
    resume: bool = True,
    difficulty_filter: str = None
):

    
    print("="*80)
    print("运行Agent - WebWalkerQA Dataset")
    print("="*80)
    
           
    print(f"\n加载数据集: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 按难度级别过滤
    if difficulty_filter:
        original_count = len(dataset)
        dataset = [item for item in dataset if item.get('difficulty_level', '').lower() == difficulty_filter.lower()]
        print(f"按难度级别 '{difficulty_filter}' 过滤: {original_count} -> {len(dataset)} 个样本")
        if len(dataset) == 0:
            print(f"警告: 没有找到难度级别为 '{difficulty_filter}' 的样本")
            # 重新加载数据集以获取所有难度级别
            with open(dataset_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            available_difficulties = set(item.get('difficulty_level', 'unknown') for item in all_data)
            print(f"可用的难度级别: {available_difficulties}")
            return None
    
           
    if max_examples > 0:
        dataset = dataset[start_idx:start_idx + max_examples]
    else:
        dataset = dataset[start_idx:]
    
    total = len(dataset)
    print(f"处理样本数: {total}")
    print(f"起始索引: {start_idx}")
    if difficulty_filter:
        print(f"难度级别: {difficulty_filter}")
    
            
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        difficulty_suffix = f"_{difficulty_filter}" if difficulty_filter else ""
        output_file = f'outputs/webwalker/webwalker_results{difficulty_suffix}_{timestamp}.json'
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
                      
    results = []
    processed_questions = set()
    resume_from_idx = 0
    
    if resume and output_path.exists():
        print(f"\n检测到已有结果文件: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            for item in results:
                processed_questions.add(item.get('Question'))
            
            resume_from_idx = len(results)
            print(f"已完成 {resume_from_idx} 个样本，将从第 {resume_from_idx + 1} 个继续")
            
            if resume_from_idx >= total:
                print(f"所有样本已处理完成！")
                return output_file
                
        except Exception as e:
            print(f"读取已有结果失败: {e}")
            print("将从头开始处理")
            results = []
            processed_questions = set()
            resume_from_idx = 0
    
    success_count = sum(1 for r in results if r.get('Success', False))
    
    print(f"\n{'='*80}")
    print("开始处理")
    print(f"{'='*80}\n")
    
             
    for idx in range(resume_from_idx, total):
        example = dataset[idx]
        question = example.get('Question', '')
        
        if question in processed_questions:
            print(f"跳过已处理的样本 {idx + 1}")
            continue
        
        answer = example.get('answer', '')
        domain = example.get('domain', '')
        difficulty = example.get('difficulty_level', '')
        
        print(f"\n[{idx + 1}/{total}] 处理样本 {start_idx + idx + 1}")
        print(f"问题: {question[:100]}...")
        print(f"领域: {domain}, 难度: {difficulty}")
        
        try:
                     
            result = await run_single_question(
                question=question,
                config_path=config_path,
                max_iterations=max_iterations,
                max_context_tokens=30000
            )
            
                  
            pred_answer = extract_answer_fn(result['answer'], mode='qa')
            
                  
            result_item = {
                'id': start_idx + idx + 1,
                'Question': question,
                'answer': answer,
                'domain': domain,
                'difficulty_level': difficulty,
                'type': example.get('type', ''),
                'source_website': example.get('source_website', []),
                'root_url': example.get('root_url', ''),
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
            print(f"\n✗ 处理样本 {start_idx + idx + 1} 时出错: {e}")
            import traceback
            traceback.print_exc()
            
            result_item = {
                'id': start_idx + idx + 1,
                'Question': question,
                'answer': answer,
                'domain': domain,
                'difficulty_level': difficulty,
                'type': example.get('type', ''),
                'source_website': example.get('source_website', []),
                'root_url': example.get('root_url', ''),
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


                                                      

async def llm_evaluate_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    output: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3
) -> tuple:

    
              
    if len(output) > 5000:
        output = "...(前面的推理过程省略)...\n\n" + output[-5000:]
    
    prompt = f"""你是一个专业的答案评估助手。请判断Agent的预测答案是否正确。

# 问题
{question}

# 标准答案
{labeled_answer}

# Agent的完整推理过程
{output}

# Agent的最终答案
{pred_answer}

# 你的任务
判断Agent的答案是否与标准答案等价。请特别注意：

1. **信息完整性** - Agent的答案是否包含了标准答案的所有关键信息
2. **事实准确性** - 日期、数字、名称等是否准确
3. **格式灵活性** - 忽略格式差异，关注实质内容

**重要**: 如果Agent的推理过程显示它理解了问题并得出了本质上正确的答案，即使表述方式不同也应判断为Correct。

# 回答格式
只回答"Correct"或"Incorrect"，不要有任何解释。"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                response_text = response.choices[0].message.content.strip()
                is_correct = "correct" in response_text.lower() and "incorrect" not in response_text.lower()
                return is_correct, response_text
                
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"\nLLM评估错误: {e}")
                return False, "Error"
            await asyncio.sleep(2 * (attempt + 1))
    
    return False, "Error"


async def batch_llm_evaluate(
    questions: list,
    labeled_answers: list,
    outputs: list,
    pred_answers: list,
    api_base: str,
    model_name: str,
    api_key: str,
    concurrent_limit: int = 5
) -> list:

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    semaphore = asyncio.Semaphore(concurrent_limit)

    async def evaluate_with_index(idx: int, q: str, l: str, o: str, p: str):
        result = await llm_evaluate_single(client, q, l, o, p, model_name, semaphore)
        return idx, result

    tasks = [
        evaluate_with_index(i, q, l, o, p)
        for i, (q, l, o, p) in enumerate(zip(questions, labeled_answers, outputs, pred_answers))
    ]

    indexed_results = []
    with tqdm(total=len(tasks), desc="LLM评估", ncols=100) as pbar:
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            indexed_results.append((idx, result))
            pbar.update(1)

    indexed_results.sort(key=lambda x: x[0])
    results = [r for _, r in indexed_results]

    await client.close()
    return results


def calculate_traditional_metrics(pred_answer: str, labeled_answer: str) -> dict:

    pred_norm = normalize_answer_qa(pred_answer)
    label_norm = normalize_answer_qa(labeled_answer)
    
                 
    em = int(pred_norm == label_norm)
    
                     
    acc = int(label_norm in pred_norm)
    
              
    pred_tokens = pred_norm.split()
    label_tokens = label_norm.split()
    
    if not pred_tokens or not label_tokens:
        f1 = 0.0
    else:
        common = set(pred_tokens) & set(label_tokens)
        if not common:
            f1 = 0.0
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(label_tokens)
            f1 = 2 * precision * recall / (precision + recall)
    
                      
    math_equal = is_equiv(pred_answer, labeled_answer)
    
    return {
        'acc': acc,
        'em': em,
        'f1': f1,
        'math_equal': int(math_equal)
    }


async def evaluate_webwalker_results(
    results_file: str,
    config_path: str = 'config/config.yaml',
    output_suffix: str = '_evaluated'
):

    
    print("="*80)
    print("评估WebWalkerQA结果")
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
    
    print(f"评估模型: {model_name}")
    
          
    print(f"\n加载结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"加载了 {len(results)} 个样本")
    
          
    questions = []
    labeled_answers = []
    outputs = []
    pred_answers = []
    valid_indices = []
    
    for i, item in enumerate(results):
        question = item.get('Question', '')
        labeled = item.get('answer', '')
        output = item.get('Output', '')
        predicted = item.get('Pred_Answer', '')
        
        if question and labeled and predicted and output:
            questions.append(question)
            labeled_answers.append(labeled)
            outputs.append(output)
            pred_answers.append(predicted)
            valid_indices.append(i)
    
    print(f"有效评估样本: {len(questions)}")
    
            
    print("\n计算传统指标...")
    for idx in tqdm(valid_indices, desc="传统指标", ncols=100):
        item = results[idx]
        pred = item.get('Pred_Answer', '')
        label = item.get('answer', '')
        
        metrics = calculate_traditional_metrics(pred, label)
        
        if 'Metrics' not in item:
            item['Metrics'] = {}
        
        item['Metrics'].update(metrics)
    
             
    print("\n运行LLM评估...")
    llm_results = await batch_llm_evaluate(
        questions=questions,
        labeled_answers=labeled_answers,
        outputs=outputs,
        pred_answers=pred_answers,
        api_base=api_base,
        model_name=model_name,
        api_key=api_key,
        concurrent_limit=5
    )
    
          
    for idx, (is_correct, response) in zip(valid_indices, llm_results):
        item = results[idx]
        item['Metrics']['llm_equal'] = int(is_correct)
        item['Metrics']['llm_response'] = response
    
            
    total = len(results)
    acc_sum = sum(item.get('Metrics', {}).get('acc', 0) for item in results)
    em_sum = sum(item.get('Metrics', {}).get('em', 0) for item in results)
    f1_sum = sum(item.get('Metrics', {}).get('f1', 0) for item in results)
    math_sum = sum(item.get('Metrics', {}).get('math_equal', 0) for item in results)
    llm_sum = sum(item.get('Metrics', {}).get('llm_equal', 0) for item in results)
    
    overall_metrics = {
        'total': total,
        'acc': acc_sum / total,
        'em': em_sum / total,
        'f1': f1_sum / total,
        'math_equal': math_sum / total,
        'llm_equal': llm_sum / total
    }
    
             
    difficulty_metrics = {}
    for difficulty in set(item.get('difficulty_level', 'unknown') for item in results):
        diff_items = [item for item in results if item.get('difficulty_level') == difficulty]
        if diff_items:
            diff_total = len(diff_items)
            difficulty_metrics[difficulty] = {
                'total': diff_total,
                'acc': sum(item.get('Metrics', {}).get('acc', 0) for item in diff_items) / diff_total,
                'em': sum(item.get('Metrics', {}).get('em', 0) for item in diff_items) / diff_total,
                'f1': sum(item.get('Metrics', {}).get('f1', 0) for item in diff_items) / diff_total,
                'llm_equal': sum(item.get('Metrics', {}).get('llm_equal', 0) for item in diff_items) / diff_total
            }
    
    overall_metrics['by_difficulty'] = difficulty_metrics
    
           
    domain_metrics = {}
    for domain in set(item.get('domain', 'unknown') for item in results):
        domain_items = [item for item in results if item.get('domain') == domain]
        if domain_items:
            domain_total = len(domain_items)
            domain_metrics[domain] = {
                'total': domain_total,
                'llm_equal': sum(item.get('Metrics', {}).get('llm_equal', 0) for item in domain_items) / domain_total
            }
    
    overall_metrics['by_domain'] = domain_metrics
    
          
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"\n总样本数: {total}")
    print(f"\n总体指标:")
    print(f"  acc:        {overall_metrics['acc']:.2%} ({acc_sum}/{total})")
    print(f"  em:         {overall_metrics['em']:.2%} ({em_sum}/{total})")
    print(f"  f1:         {overall_metrics['f1']:.4f}")
    print(f"  math_equal: {overall_metrics['math_equal']:.2%} ({math_sum}/{total})")
    print(f"  llm_equal:  {overall_metrics['llm_equal']:.2%} ({llm_sum}/{total}) <- 最可靠")
    
    print(f"\n按难度级别:")
    for difficulty, metrics in difficulty_metrics.items():
        print(f"  {difficulty}:")
        print(f"    总数: {metrics['total']}")
        print(f"    llm_equal: {metrics['llm_equal']:.2%}")
    
    print(f"\n按领域:")
    for domain, metrics in domain_metrics.items():
        print(f"  {domain}: {metrics['llm_equal']:.2%} ({metrics['total']}个)")
    
    print("="*80)
    
          
    results_path = Path(results_file)
    base_name = results_path.stem
    output_dir = results_path.parent
    
    detailed_file = output_dir / f"{base_name}{output_suffix}.json"
    print(f"\n保存详细结果至: {detailed_file}")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    overall_file = output_dir / f"{base_name}{output_suffix}.overall.json"
    print(f"保存总体指标至: {overall_file}")
    with open(overall_file, 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("评估完成!")
    print("="*80)
    
    return detailed_file, overall_file


                                                

def find_latest_result_file(output_dir: str, difficulty: str = None) -> str:
    """找到指定难度的最新结果文件"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    if difficulty:
        pattern = f"webwalker_results_{difficulty}_*.json"
    else:
        pattern = "webwalker_results_*.json"

    # 排除已评估的文件
    result_files = [
        f for f in output_path.glob(pattern)
        if '_evaluated' not in f.stem
    ]

    if not result_files:
        return None

    # 按修改时间排序，返回最新的
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


def main():
    parser = argparse.ArgumentParser(
        description="WebWalkerQA数据集评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--eval_only', action='store_true', help='只评估已有结果')
    mode_group.add_argument('--run_only', action='store_true', help='只运行agent，不评估')

    parser.add_argument('--dataset', type=str, default='data/WebWalkerQA/test.json', help='数据集路径')
    parser.add_argument('--max_examples', type=int, default=-1, help='最大处理样本数')
    parser.add_argument('--start_idx', type=int, default=0, help='起始索引')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--max_iterations', type=int, default=25, help='最大迭代次数')
    parser.add_argument('--output_dir', type=str, default='outputs/webwalker', help='输出目录')
    parser.add_argument('--output_name', type=str, default=None, help='输出文件名')
    parser.add_argument('--results', type=str, help='要评估的结果文件')
    parser.add_argument('--output_suffix', type=str, default='_evaluated', help='评估输出文件后缀')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'],
                       help='只处理指定难度级别的样本 (easy/medium/hard)')
    parser.add_argument('--resume_latest', action='store_true',
                       help='自动找到最新的结果文件并从断点继续')

    args = parser.parse_args()
    
    if args.eval_only and not args.results:
        parser.error("--eval_only 需要 --results 参数")

    # 处理 --resume_latest：自动找到最新的结果文件
    if args.resume_latest:
        latest_file = find_latest_result_file(args.output_dir, args.difficulty)
        if latest_file:
            print(f"\n找到最新的结果文件: {latest_file}")
            output_file = Path(latest_file)
            args.output_name = output_file.name
        else:
            print(f"\n未找到已有结果文件，将创建新文件")
            args.resume_latest = False  # 没找到就当作新任务

    if args.output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        difficulty_suffix = f"_{args.difficulty}" if args.difficulty else ""
        args.output_name = f'webwalker_results{difficulty_suffix}_{timestamp}.json'

    output_file = Path(args.output_dir) / args.output_name
    
    print("\n" + "="*80)
    print("WebWalkerQA 数据集评估脚本")
    print("="*80)
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    if args.eval_only:
        asyncio.run(evaluate_webwalker_results(
            results_file=args.results,
            config_path=args.config,
            output_suffix=args.output_suffix
        ))
    elif args.run_only:
        results_file = asyncio.run(run_agent_on_webwalker(
            dataset_path=args.dataset,
            output_file=str(output_file),
            max_examples=args.max_examples,
            start_idx=args.start_idx,
            config_path=args.config,
            max_iterations=args.max_iterations,
            difficulty_filter=args.difficulty
        ))
        if results_file is None:
            return
    else:
        results_file = asyncio.run(run_agent_on_webwalker(
            dataset_path=args.dataset,
            output_file=str(output_file),
            max_examples=args.max_examples,
            start_idx=args.start_idx,
            config_path=args.config,
            max_iterations=args.max_iterations,
            difficulty_filter=args.difficulty
        ))
        if results_file is None:
            return
        
        asyncio.run(evaluate_webwalker_results(
            results_file=results_file,
            config_path=args.config,
            output_suffix=args.output_suffix
        ))


if __name__ == "__main__":
    main()
