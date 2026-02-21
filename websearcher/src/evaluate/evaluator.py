


   

import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import string
import asyncio
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import sys
import os

                                          
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .metrics import (
    extract_answer_fn,
    normalize_answer_qa,
    normalize_answer
)


def _extract_first_number(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _nearly_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def _unit_scaled_match(question: Optional[str], pred: str, truth: str) -> bool:
    if not question:
        return False
    q = question.lower()
    if "thousand" not in q and "thousands" not in q and "千" not in q:
        return False
    pred_num = _extract_first_number(pred)
    truth_num = _extract_first_number(truth)
    if pred_num is None or truth_num is None:
        return False
    if _nearly_equal(pred_num, truth_num * 1000) or _nearly_equal(pred_num / 1000, truth_num):
        return True
    return False
from utils.math_equivalence import is_equiv


async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
    extract_answer: bool = False,
) -> tuple:















       
    if extract_answer:
        prompt = f"""You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Predicted Answer: {pred_answer}

Are these answers equivalent? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""
    else:
        prompt = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = chat_response.choices[0].message.content.strip()
                llm_judge = is_equiv(pred_answer, labeled_answer) or\
                    response_text.lower() == "correct" and\
                    not ("incorrect" in response_text.lower() or\
                         "wrong" in response_text.lower() or\
                         "not correct" in response_text.lower())
                return llm_judge, response_text
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                return is_equiv(pred_answer, labeled_answer), "Error"
            await asyncio.sleep(1 * (attempt + 1))

    return is_equiv(pred_answer, labeled_answer), "Error"


async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str],
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 3,                                      
    extract_answer: bool = False
) -> List[tuple]:















       
    if model_name is None:
        model_name = "qwen-plus"

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )

    semaphore = asyncio.Semaphore(concurrent_limit)

    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
            extract_answer=extract_answer
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result

        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)

    return results


def evaluate_predictions(
    output: str,
    labeled_answer,
    mode: str = 'math',
    use_llm: bool = False,
    question: str = None,
    extract_answer: bool = False
) -> tuple:













       
    final_metric = {
        "is_valid_answer": False,
        "acc": 0,
        "em": 0,
        "f1": 0,
        'math_equal': 0,
        'llm_equal': 0
    }

    pred_answer = extract_answer_fn(output, mode=mode, extract_answer=extract_answer)
    pred_answer_new = pred_answer

    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    else:
                                                                
        pred_answer_new = '\n'.join(output.replace("\n\n", "\n").strip().split('\n')[-5:])

    if mode in ['qa']:
        normalized_pred_answer = normalize_answer_qa(pred_answer_new)

                                          
        if not isinstance(labeled_answer, list):
            labeled_answer = [labeled_answer]

        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue                               
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    elif mode in ['math', 'choose']:
        normalized_pred_answer = normalize_answer(pred_answer_new)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)

        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1
        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

        if _unit_scaled_match(question, pred_answer_new, labeled_answer):
            final_metric["em"] = 1
            final_metric["acc"] = 1
            final_metric["f1"] = 1.0
            final_metric["math_equal"] = 1

                                               
        if use_llm and question is not None:
            final_metric["llm_equal"] = 0                                  

    return final_metric, pred_answer


def run_evaluation(
    filtered_data: List[Dict],
    input_list: List[str],
    output_list: List[str],
    task_type: str,
    output_dir: str,
    output_metrics_path: str,
    output_metrics_overall_path: str,
    use_llm: bool = False,
    extract_answer: bool = False,
    domain_fields: List[str] = None,
    api_base_url: str = None,
    model_name: str = None
):
















       
                                          
    domain_metrics = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'em': [],
        'acc': [],
        'f1': [],
        'math_equal': [],
        'llm_equal': [],
        'pass@1': []
    })

                                             
    def get_domain(item):
        if domain_fields:
            for field in domain_fields:
                if field in item and item[field] is not None:
                    return item[field]
        return 'Unknown'

    if task_type == 'code':
                                                                 
        num_valid_answer = 0

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text

            if item['Output'] == '':
                item['Pred_Answer'] = ''
                item['Question'] = input_prompt
                item['Metrics'] = {'pass@1': 0}
                continue

            pred_code = extract_answer_fn(item['Output'], mode='codegen', extract_answer=extract_answer)
            if pred_code != '':
                num_valid_answer += 1

            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt
            item['Metrics'] = {'pass@1': 0.0}               

        overall_metrics = {
            'pass@1': 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        }

    elif task_type in ['math', 'choose', 'qa']:
                            
        avg_em, avg_acc, avg_f1, avg_math, avg_llm = [], [], [], [], []
        num_valid_answer = 0

                                                      
        questions_for_llm = []
        labeled_answers_for_llm = []
        pred_answers_for_llm = []
        items_for_llm = []

        for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list), total=len(input_list), desc="Evaluating"):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text

            if item['Output'] == '':
                item['Pred_Answer'] = ''
                item['Question'] = input_prompt
                item['Metrics'] = {
                    'em': 0,
                    'acc': 0,
                    'f1': 0,
                    'math_equal': 0,
                    'llm_equal': 0 if use_llm else None
                }
                avg_em.append(0)
                avg_acc.append(0)
                avg_f1.append(0)
                avg_math.append(0)
                if use_llm:
                    avg_llm.append(0)
                continue

                                                  
            labeled_answer = item.get('answer', '')
            if 'Correct Choice' in item and item['Correct Choice'] is not None:
                labeled_answer = item['Correct Choice']
            elif 'answer_letter' in item and item['answer_letter'] is not None:
                labeled_answer = item['answer_letter']

            metric, pred_answer = evaluate_predictions(
                output=item['Output'],
                labeled_answer=labeled_answer,
                mode=task_type,
                use_llm=use_llm,
                question=input_prompt,
                extract_answer=extract_answer
            )

            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Question'] = input_prompt

                                                 
            if use_llm:
                questions_for_llm.append(input_prompt)
                labeled_answers_for_llm.append(labeled_answer)
                pred_answers_for_llm.append(pred_answer)
                items_for_llm.append(item)

                                                            
            my_method_valid = (pred_answer != '')

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

                                                
        if use_llm and questions_for_llm:
            print("\nRunning LLM-based evaluation...")
                                                                                
            import asyncio
            import sys

                                                
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                llm_results = loop.run_until_complete(llm_evaluate_equivalence_batch(
                    questions=questions_for_llm,
                    labeled_answers=labeled_answers_for_llm,
                    pred_answers=pred_answers_for_llm,
                    extract_answer=extract_answer,
                    api_base_url=api_base_url,
                    model_name=model_name
                ))
            finally:
                try:
                    loop.close()
                except:
                    pass

                                             
            for item, (llm_result, llm_response) in zip(items_for_llm, llm_results):
                item['Metrics']['llm_equal'] = int(llm_result)
                item['Metrics']['llm_response'] = llm_response
                avg_llm.append(int(llm_result))

                                 
        overall_metrics = {
            'em': float(np.mean(avg_em)) if len(avg_em) > 0 else 0.0,
            'acc': float(np.mean(avg_acc)) if len(avg_acc) > 0 else 0.0,
            'f1': float(np.mean(avg_f1)) if len(avg_f1) > 0 else 0.0,
            'math_equal': float(np.mean(avg_math)) if len(avg_math) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        }

                                                
        if len(avg_llm) > 0:
            overall_metrics['llm_equal'] = float(np.mean(avg_llm))

                                         
        for item in filtered_data:
            if 'Metrics' not in item:
                continue
            metric = item['Metrics']
            domain = get_domain(item)
            domain_metrics[domain]['total'] += 1
            domain_metrics[domain]['em'].append(metric.get('em', 0))
            domain_metrics[domain]['acc'].append(metric.get('acc', 0))
            domain_metrics[domain]['f1'].append(metric.get('f1', 0))
            domain_metrics[domain]['math_equal'].append(metric.get('math_equal', 0))
            if 'llm_equal' in metric and metric['llm_equal'] is not None:
                domain_metrics[domain]['llm_equal'].append(metric['llm_equal'])

                                       
    domain_metrics_final = {}
    for domain, metrics in domain_metrics.items():
        domain_metrics_final[domain] = {
            'total': metrics['total'],
            'em': float(np.mean(metrics['em'])) if len(metrics['em']) > 0 else 0.0,
            'acc': float(np.mean(metrics['acc'])) if len(metrics['acc']) > 0 else 0.0,
            'f1': float(np.mean(metrics['f1'])) if len(metrics['f1']) > 0 else 0.0,
            'math_equal': float(np.mean(metrics['math_equal'])) if len(metrics['math_equal']) > 0 else 0.0,
        }
        if metrics['llm_equal']:
            domain_metrics_final[domain]['llm_equal'] = float(np.mean(metrics['llm_equal']))
        if metrics['pass@1']:
            domain_metrics_final[domain]['pass@1'] = float(np.mean(metrics['pass@1']))

                                           
    overall_metrics['domain_metrics'] = domain_metrics_final

                           
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in overall_metrics.items():
        if key != 'domain_metrics':
            print(f"{key}: {value}")
    print("="*80)

                                                                             
    import os
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=2, ensure_ascii=False)

    print(f"\n📁 Metrics saved to:")
    print(f"   Detailed: {os.path.join(output_dir, output_metrics_path)}")
    print(f"   Overall: {os.path.join(output_dir, output_metrics_overall_path)}")
