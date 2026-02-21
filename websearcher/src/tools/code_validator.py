




import re
import json
from typing import Dict, Optional, Tuple


class CodeValidator:

    
    @staticmethod
    def validate_result(
        code: str,
        result: str,
        question: str = ""
    ) -> Tuple[bool, Optional[str]]:













        warnings = []
        
                       
        if "error" in result.lower() or "exception" in result.lower():
            warnings.append("⚠️ 代码执行可能有错误")
        
                       
        simulation_match = re.search(r'num_simulations?\s*=\s*(\d+)', code)
        if simulation_match:
            num_sims = int(simulation_match.group(1))
            if num_sims < 1000:
                warnings.append(f"⚠️ 模拟次数较少 ({num_sims}),结果可能不准确。建议至少10000次")
        
                      
        prob_matches = re.findall(r'probability[:\s]+(\d+\.?\d*)', result.lower())
        for prob_str in prob_matches:
            prob = float(prob_str)
            if prob > 1.0:
                warnings.append(f"⚠️ 概率值 {prob} > 1.0,这不合理")
            elif prob < 0.0:
                warnings.append(f"⚠️ 概率值 {prob} < 0.0,这不合理")
            elif prob > 0.99:
                warnings.append(f"⚠️ 概率值 {prob} 接近1,请检查是否合理")
        
                         
        if "while" in code and "break" not in code:
                    
            if "for" not in code:                        
                warnings.append("⚠️ 代码包含while循环但没有明确的break条件,可能有逻辑错误")
        
                     
        if "platform" in code and "pop" in code:
                                   
            if code.count("if piston ==") >= 2 or code.count("elif piston ==") >= 2:
                             
                if "else:" not in code and code.count("elif") < 2:
                    warnings.append("⚠️ 状态转换逻辑可能不完整,建议检查所有分支")
        
                         
        if "platform[" in code:
            if "len(platform)" not in code:
                warnings.append("⚠️ 访问platform元素但没有检查长度,可能导致索引错误")
        
                      
        # 检查代码中是否有未初始化的变量使用
        if "=" not in code and "def" not in code:
            # 如果代码中没有赋值或函数定义，可能是片段代码
            pass
        
        # 检查是否有明显的逻辑错误模式（泛化检查）
        if "if" in code and "else" not in code:
            # 如果有很多if但没有else，可能缺少默认分支
            if_count = code.count("if ")
            elif_count = code.count("elif ")
            else_count = code.count("else:")
            if if_count + elif_count > 3 and else_count == 0:
                warnings.append("⚠️ 多个条件分支但没有else分支,考虑添加默认情况处理")
        
                    
        # 检查是否输出了所有组合/列表而不是最终答案
        if "(" in result and "," in result and ")" in result:
            # 检查是否有类似 (a, b), (c, d) 这样的多个元组输出
            tuple_pattern = r'\([^)]+\)'
            tuples = re.findall(tuple_pattern, result)
            if len(tuples) > 5:  # 如果有很多元组，可能是输出了所有组合而不是答案
                warnings.append("⚠️ 输出包含大量元组/组合,请确认是否输出了所有可能组合而不是最终答案")
        
        # 检查是否输出了中间过程而不是最终答案
        if "test" in result.lower() and "result" in result.lower():
            # 如果包含 "Test result" 等中间输出
            test_count = result.lower().count("test")
            if test_count > 3:
                warnings.append("⚠️ 输出包含大量测试结果,请确认是否输出了中间过程而不是最终答案")
        
        # 检查数值精度问题（迭代算法收敛判断）
        if "newton" in code.lower() or ("iteration" in code.lower() and "converge" not in code.lower()):
            # 检查是否有收敛判断逻辑
            if "abs" not in code and "abs(" not in code:
                warnings.append("⚠️ 迭代算法建议添加收敛判断(如检查相邻迭代值的差值是否小于阈值)")
            # 检查是否使用了合适的精度阈值
            if "1e-" not in code and "0.0001" not in code and "0.001" not in code:
                warnings.append("⚠️ 迭代算法建议设置明确的收敛阈值(如 1e-6 或 0.0001)")
        
        # 检查答案格式（通用检查）
        if "best" in result.lower() or "maximum" in result.lower() or "answer" in result.lower():
            # 提取所有数字
            numbers = re.findall(r'\b(\d+)\b', result)
            if numbers:
                # 如果结果中有很多数字，可能是输出了中间过程
                if len(numbers) > 10:
                    warnings.append("⚠️ 输出包含大量数字,请确认是否输出了中间过程而不是最终答案")
        
                
        if warnings:
            warning_msg = "\n".join(warnings)
            warning_msg += "\n\n💡 通用建议:\n"
            warning_msg += "1. 仔细检查代码逻辑,特别是边界条件和状态转换\n"
            warning_msg += "2. 添加调试输出验证中间结果\n"
            warning_msg += "3. 检查输出格式是否符合题目要求\n"
            warning_msg += "4. 对于迭代算法,确保有正确的收敛判断\n"
            warning_msg += "5. 验证最终答案是否合理(如范围、格式等)"
            return False, warning_msg
        
        return True, None
    
    @staticmethod
    def suggest_improvements(code: str) -> str:









        suggestions = []
        
                
        if "num_simulations" in code:
            sim_match = re.search(r'num_simulations?\s*=\s*(\d+)', code)
            if sim_match and int(sim_match.group(1)) < 10000:
                suggestions.append("• 增加模拟次数到至少10000次以提高准确性")
        
                   
        if "print" not in code or code.count("print") < 3:
            suggestions.append("• 添加更多调试输出,打印中间状态以验证逻辑")
        
                   
        if "test" not in code.lower() and "assert" not in code:
            suggestions.append("• 添加简单的测试用例验证代码逻辑")
        
                 
        if code.count("#") < 5:
            suggestions.append("• 添加更多注释说明每个步骤的逻辑")
        
        if suggestions:
            return "代码改进建议:\n" + "\n".join(suggestions)
        
        return ""


def validate_python_result(
    code: str,
    result: str,
    question: str = ""
) -> Dict:











    validator = CodeValidator()
    is_valid, warning = validator.validate_result(code, result, question)
    
    return {
        "is_valid": is_valid,
        "warning": warning,
        "suggestions": validator.suggest_improvements(code) if not is_valid else None
    }
