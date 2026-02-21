




import re
import string
from collections import Counter
from typing import Union, List
import sys
import os

                                          
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_equivalence import is_equiv as math_is_equiv
from utils.text_utils import extract_answer as unified_extract_answer, normalize_answer_qa


def extract_answer_fn(output: str, mode: str = 'qa', extract_answer: bool = False) -> str:












                                                                   
    if not extract_answer and mode not in ['infogen', 'summary', 'research', 'math', 'choose']:
        if mode == 'qa':
                                          
            extracted = unified_extract_answer(output, mode='qa')
            if extracted:
                return extracted
                                         
            return output.strip()
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer

                            
    extracted = unified_extract_answer(output, mode=mode)
    if extracted:
        return extracted

                                                                    
    if mode in ['infogen', 'summary', 'research']:
        lines = output.strip().replace("\n\n", "\n").split('\n')
        return '\n'.join(lines[-5:])[:2500]

    return output.strip()


def normalize_answer(text: str) -> str:









    text = text.lower()
    text = " ".join(text.strip().split())
    return text


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:












    return math_is_equiv(str1, str2, verbose=verbose)
