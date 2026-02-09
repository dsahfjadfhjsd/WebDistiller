















   
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


                                                           

async def run_agent_on_dataset(
    dataset_path: str,
    output_file: str,
    max_examples: int = -1,
    start_idx: int = 0,
    config_path: str = 'config/config.yaml',
    max_iterations: int = 20,
    resume: bool = True
):
                              
    
    print("="*80)
    print("运行Agent")
    print("="*80)
    
           
    print(f"\n加载数据集: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
              
    if isinstance(dataset, dict):
        dataset = list(dataset.values())
    
           
    if max_examples > 0:
        dataset = dataset[start_idx:start_idx + max_examples]
    else:
        dataset = dataset[start_idx:]
    
    total = len(dataset)
    print(f"处理样本数: {total}")
    print(f"起始索引: {start_idx}")
    
            
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
        example_num = start_idx + idx + 1
        
                  
        if example_num in processed_ids:
            print(f"跳过已处理的样本 {example_num}")
            continue
        
              
        question = example.get('Question', example.get('question', ''))
        
        print(f"\n[{idx + 1}/{total}] 处理样本 {example_num}")
        print(f"问题: {question[:100]}...")
        
        try:
                     
            result = await run_single_question(
                question=question,
                config_path=config_path,
                max_iterations=max_iterations,
                max_context_tokens=30000
            )
            
                  
            pred_answer = extract_answer_fn(result['answer'], mode='qa')
            
                  
            result_item = {
                'id': example_num,
                'task_id': example.get('task_id', ''),
                'Question': question,
                'answer': example.get('answer', ''),
                'Level': example.get('Level', 0),
                'Annotator_Metadata': example.get('Annotator_Metadata', {}),
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
            print(f"\n✗ 处理样本 {example_num} 时出错: {e}")
            import traceback
            traceback.print_exc()
            
                    
            result_item = {
                'id': example_num,
                'task_id': example.get('task_id', ''),
                'Question': question,
                'answer': example.get('answer', ''),
                'Level': example.get('Level', 0),
                'Annotator_Metadata': example.get('Annotator_Metadata', {}),
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

1. **数字和单位的等价性**
   - "17千时" = "17000时" = "17000" (千=1000)
   - "1.5万" = "15000" = "15,000" (万=10000)
   - "50%" = "0.5" = "50/100" (百分比和小数)
   - "1/2" = "0.5" = "50%"

2. **单位转换**
   - "1km" = "1000m" = "0.621 miles"
   - "1小时" = "60分钟" = "3600秒"

3. **日期时间格式**
   - "2024-01-15" = "January 15, 2024" = "15/01/2024"
   - "14:30" = "2:30 PM" = "下午2点30分"

4. **文本格式**
   - 大小写不敏感: "Paris" = "paris" = "PARIS"
   - 空格和标点: "New York" = "New York " = "New-York"

5. **合理舍入**
   - "3.14" ≈ "3.14159" (π的近似值)
   - "2.5" ≈ "2.50" = "2.500"

**重要**: 如果Agent的推理过程显示它理解了问题并得出了本质上正确的答案，即使格式不同也应判断为Correct。

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


async def evaluate_results(
    results_file: str,
    config_path: str = 'config/config.yaml',
    output_suffix: str = '_evaluated'
):
                
    
    print("="*80)
    print("评估结果")
    print("="*80)
    
          
    print("\n加载配置...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
              
    eval_model = config.get('model', {}).get('evaluation_model', {})
    if not eval_model:
        print("警告: 配置中没有evaluation_model，使用auxiliary_model")
        eval_model = config.get('model', {}).get('auxiliary_model', {})
    
    api_base = eval_model.get('api_base', 'https://api.openai.com/v1')
    model_name = eval_model.get('name', 'deepseek-v3.2')
    api_key = eval_model.get('api_key', 'empty')
    
    print(f"评估模型: {model_name}")
    print(f"API Base: {api_base}")
    
          
    print(f"\n加载结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"加载了 {len(results)} 个样本")
    
          
    print("\n准备评估数据...")
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
    print(f"评估模型: {model_name}")
    print(f"并发限制: 5")
    print(f"这可能需要几分钟...\n")
    
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
    
          
    print("\n更新LLM评估结果...")
    for idx, (is_correct, response) in zip(valid_indices, llm_results):
        item = results[idx]
        item['Metrics']['llm_equal'] = int(is_correct)
        item['Metrics']['llm_response'] = response
    
            
    print("\n计算总体指标...")
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
    
             
    level_metrics = {}
    for level in [1, 2, 3]:
        level_items = [item for item in results if item.get('Level') == level]
        if level_items:
            level_total = len(level_items)
            level_metrics[f'Level_{level}'] = {
                'total': level_total,
                'acc': sum(item.get('Metrics', {}).get('acc', 0) for item in level_items) / level_total,
                'em': sum(item.get('Metrics', {}).get('em', 0) for item in level_items) / level_total,
                'f1': sum(item.get('Metrics', {}).get('f1', 0) for item in level_items) / level_total,
                'math_equal': sum(item.get('Metrics', {}).get('math_equal', 0) for item in level_items) / level_total,
                'llm_equal': sum(item.get('Metrics', {}).get('llm_equal', 0) for item in level_items) / level_total
            }
    
    overall_metrics['by_level'] = level_metrics
    
          
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
    for level_name, metrics in level_metrics.items():
        print(f"  {level_name}:")
        print(f"    总数: {metrics['total']}")
        print(f"    llm_equal: {metrics['llm_equal']:.2%}")
    
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


                                                

def main():
    parser = argparse.ArgumentParser(
        description="WebSearcher统一评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程（运行agent + 评估）
  python evaluate.py --dataset data/GAIA/dev.json --max_examples 10
  
  # 只评估已有结果
  python evaluate.py --eval_only --results outputs/gaia/gaia_results.json
  
  # 只运行agent（不评估）
  python evaluate.py --dataset data/GAIA/dev.json --run_only
  
  # 自定义参数
  python evaluate.py --dataset data/GAIA/dev.json --max_examples 50 --start_idx 10 --max_iterations 30
        """
    )
    
          
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--eval_only', action='store_true',
                           help='只评估已有结果（需要--results）')
    mode_group.add_argument('--run_only', action='store_true',
                           help='只运行agent，不评估')
    
           
    parser.add_argument('--dataset', type=str,
                       default='data/GAIA/dev.json',
                       help='数据集路径（默认: data/GAIA/dev.json）')
    parser.add_argument('--max_examples', type=int, default=-1,
                       help='最大处理样本数（-1表示全部，默认: -1）')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='起始索引（默认: 0）')
    
             
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径（默认: config/config.yaml）')
    parser.add_argument('--max_iterations', type=int, default=None,
                       help='每个问题的最大迭代次数；不填则使用难度估计的建议值（例如 HARD=30）')
    
          
    parser.add_argument('--output_dir', type=str, default='outputs/gaia',
                       help='输出目录（默认: outputs/gaia）')
    parser.add_argument('--output_name', type=str, default=None,
                       help='输出文件名（默认: gaia_results_TIMESTAMP.json）')
    
          
    parser.add_argument('--results', type=str,
                       help='要评估的结果文件（--eval_only时必需）')
    parser.add_argument('--output_suffix', type=str, default='_evaluated',
                       help='评估输出文件后缀（默认: _evaluated）')
    
    args = parser.parse_args()
    
          
    if args.eval_only and not args.results:
        parser.error("--eval_only 需要 --results 参数")
    
             
    if args.output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_name = f'gaia_results_{timestamp}.json'
    
    output_file = Path(args.output_dir) / args.output_name
    
    print("\n" + "="*80)
    print("WebSearcher 统一评估脚本")
    print("="*80)
    
                
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
          
    if args.eval_only:
             
        print(f"模式: 只评估")
        print(f"结果文件: {args.results}")
        print("="*80 + "\n")
        
        asyncio.run(evaluate_results(
            results_file=args.results,
            config_path=args.config,
            output_suffix=args.output_suffix
        ))
        
    elif args.run_only:
                  
        print(f"模式: 只运行Agent")
        print(f"数据集: {args.dataset}")
        print(f"输出: {output_file}")
        print(f"最大样本数: {args.max_examples if args.max_examples > 0 else '全部'}")
        print(f"起始索引: {args.start_idx}")
        print(f"最大迭代: {args.max_iterations}")
        print("="*80 + "\n")
        
        asyncio.run(run_agent_on_dataset(
            dataset_path=args.dataset,
            output_file=str(output_file),
            max_examples=args.max_examples,
            start_idx=args.start_idx,
            config_path=args.config,
            max_iterations=args.max_iterations
        ))
        
    else:
              
        print(f"模式: 完整流程（运行Agent + 评估）")
        print(f"数据集: {args.dataset}")
        print(f"输出: {output_file}")
        print(f"最大样本数: {args.max_examples if args.max_examples > 0 else '全部'}")
        print(f"起始索引: {args.start_idx}")
        print(f"最大迭代: {args.max_iterations}")
        print("="*80 + "\n")
        
                      
        results_file = asyncio.run(run_agent_on_dataset(
            dataset_path=args.dataset,
            output_file=str(output_file),
            max_examples=args.max_examples,
            start_idx=args.start_idx,
            config_path=args.config,
            max_iterations=args.max_iterations
        ))
        
                 
        asyncio.run(evaluate_results(
            results_file=results_file,
            config_path=args.config,
            output_suffix=args.output_suffix
        ))
        
        print("\n" + "="*80)
        print("完整流程结束")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
