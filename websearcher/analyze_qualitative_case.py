"""
分析输出数据，找到最有代表性的定性分析案例
用于Figure 4的绘制
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

def find_best_qualitative_case(
    output_dir: str,
    min_iterations: int = 3,
    min_tool_calls: int = 2,
    require_memory_fold: bool = True
) -> Optional[Dict]:
    """
    找到最有代表性的案例：
    - 多步骤推理（至少3步）
    - 多次工具调用（至少2次）
    - 有memory folding（如果要求）
    - 成功回答
    - Level 2或3（复杂任务）
    """
    best_case = None
    best_score = 0
    
    output_path = Path(output_dir)
    json_files = list(output_path.glob("*.json"))
    json_files = [f for f in json_files if "evaluated" not in f.name and "overall" not in f.name]
    
    print(f"找到 {len(json_files)} 个输出文件")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
                
            for item in data:
                # 筛选条件
                if not item.get('Success', False):
                    continue
                    
                level = item.get('Level', 0)
                if level < 2:  # 只要Level 2或3
                    continue
                    
                iterations = item.get('Iterations', 0)
                tool_calls = item.get('Tool_Calls', 0)
                memory_folds = item.get('Memory_Folds', 0)
                
                if iterations < min_iterations:
                    continue
                if tool_calls < min_tool_calls:
                    continue
                if require_memory_fold and memory_folds == 0:
                    continue
                
                # 计算分数：综合考虑复杂度
                score = (
                    iterations * 2 +
                    tool_calls * 3 +
                    memory_folds * 5 +
                    level * 10 +
                    (1 if item.get('WebExplorer') else 0) * 5
                )
                
                if score > best_score:
                    best_score = score
                    best_case = item.copy()
                    best_case['source_file'] = str(json_file)
                    
        except Exception as e:
            print(f"读取 {json_file} 时出错: {e}")
            continue
    
    return best_case

def extract_reasoning_details(case: Dict) -> Dict:
    """从案例中提取推理细节"""
    output = case.get('Output', '')
    question = case.get('Question', '')
    
    # 尝试从Output中提取推理步骤
    # 这里需要根据实际输出格式来解析
    details = {
        'question': question,
        'answer': case.get('Pred_Answer', ''),
        'ground_truth': case.get('answer', ''),
        'level': case.get('Level', 0),
        'iterations': case.get('Iterations', 0),
        'tool_calls': case.get('Tool_Calls', 0),
        'memory_folds': case.get('Memory_Folds', 0),
        'total_tokens': case.get('Total_Tokens', 0),
        'full_output': output,
        'web_explorer': case.get('WebExplorer', [])
    }
    
    return details

if __name__ == "__main__":
    # 分析GAIA数据
    print("=" * 80)
    print("分析GAIA输出数据")
    print("=" * 80)
    
    gaia_case = find_best_qualitative_case(
        "outputs/gaia",
        min_iterations=3,
        min_tool_calls=2,
        require_memory_fold=False  # 先不要求memory fold，看看有什么
    )
    
    if gaia_case:
        print("\n找到最佳GAIA案例:")
        print(f"  ID: {gaia_case.get('id', 'N/A')}")
        print(f"  Level: {gaia_case.get('Level', 'N/A')}")
        print(f"  Iterations: {gaia_case.get('Iterations', 'N/A')}")
        print(f"  Tool Calls: {gaia_case.get('Tool_Calls', 'N/A')}")
        print(f"  Memory Folds: {gaia_case.get('Memory_Folds', 'N/A')}")
        print(f"  Question: {gaia_case.get('Question', '')[:200]}...")
        print(f"  Answer: {gaia_case.get('Pred_Answer', 'N/A')}")
        print(f"  Ground Truth: {gaia_case.get('answer', 'N/A')}")
        print(f"  Source: {gaia_case.get('source_file', 'N/A')}")
        
        # 保存详细信息
        details = extract_reasoning_details(gaia_case)
        with open("outputs/qualitative_case_gaia.json", 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print("\n详细信息已保存到: outputs/qualitative_case_gaia.json")
    else:
        print("\n未找到符合条件的GAIA案例")
    
    # 分析GPQA数据
    print("\n" + "=" * 80)
    print("分析GPQA输出数据")
    print("=" * 80)
    
    gpqa_case = find_best_qualitative_case(
        "outputs/gpqa",
        min_iterations=3,
        min_tool_calls=2,
        require_memory_fold=False
    )
    
    if gpqa_case:
        print("\n找到最佳GPQA案例:")
        print(f"  ID: {gpqa_case.get('id', 'N/A')}")
        print(f"  Iterations: {gpqa_case.get('Iterations', 'N/A')}")
        print(f"  Tool Calls: {gpqa_case.get('Tool_Calls', 'N/A')}")
        print(f"  Memory Folds: {gpqa_case.get('Memory_Folds', 'N/A')}")
        print(f"  Question: {gpqa_case.get('Question', '')[:200]}...")
        print(f"  Answer: {gpqa_case.get('Pred_Answer', 'N/A')}")
        print(f"  Ground Truth: {gpqa_case.get('answer', 'N/A')}")
        
        details = extract_reasoning_details(gpqa_case)
        with open("outputs/qualitative_case_gpqa.json", 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print("\n详细信息已保存到: outputs/qualitative_case_gpqa.json")
    else:
        print("\n未找到符合条件的GPQA案例")
