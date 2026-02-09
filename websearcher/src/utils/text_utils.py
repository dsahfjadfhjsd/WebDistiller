



   

import re
import string
from typing import Optional, List, Tuple, Dict

                 
_BOXED_PATTERN = re.compile(r'\\boxed\{([^{}]+)\}')
_BOXED_NESTED_PATTERN = re.compile(r'\\boxed\{')
_ANSWER_PATTERNS = [
    re.compile(r'[Ff]inal\s+[Aa]nswer[:\s]+(.+?)(?:\n|$)', re.DOTALL),
    re.compile(r'[Tt]he\s+answer\s+is[:\s]+(.+?)(?:\n|$)', re.DOTALL),
    re.compile(r'[Aa]nswer[:\s]+(.+?)(?:\n|$)', re.DOTALL),
    re.compile(r'ANSWER[:\s]+(.+?)(?:\n|$)', re.DOTALL),
    re.compile(r'\*\*[Aa]nswer\*\*[:\s]+(.+?)(?:\n|$)', re.DOTALL),
]
_TEXT_WRAPPER_PATTERN = re.compile(r'\\text\{(.*?)\}')
_TOOL_RESULT_BLOCK_PATTERN = re.compile(r'<tool_call_result>.*?</tool_call_result>', re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_PATTERN = re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL | re.IGNORECASE)
_CLEAN_PATTERNS = [
    (re.compile(r'^[\"\']|[\"\']$'), ''),                 
    (re.compile(r'^\*\*|\*\*$'), ''),                        
    (re.compile(r'^`|`$'), ''),                    
]

                
_tiktoken_encoder = None


def extract_between(text: str, start: str, end: str) -> str:










       
    if not text:
        return ""
    try:
        start_idx = text.rfind(start)
        if start_idx == -1:
            return ""
        start_idx += len(start)

        end_idx = text.find(end, start_idx)
        if end_idx == -1:
                                  
            end_idx = len(text)

        return text[start_idx:end_idx].strip()
    except Exception:
        return ""


def _strip_marker_blocks(text: str, markers: Dict[str, str]) -> str:
    begin_result = markers.get("BEGIN_TOOL_RESULT")
    end_result = markers.get("END_TOOL_RESULT")
    if begin_result and end_result:
        pattern = re.compile(re.escape(begin_result) + r".*?" + re.escape(end_result), re.DOTALL)
        text = pattern.sub("", text)

    begin_call = markers.get("BEGIN_TOOL_CALL")
    end_call = markers.get("END_TOOL_CALL")
    if begin_call and end_call:
        pattern = re.compile(re.escape(begin_call) + r".*?" + re.escape(end_call), re.DOTALL)
        text = pattern.sub("", text)

    return text


def extract_answer(
    text: str,
    mode: str = 'qa',
    markers: Optional[Dict[str, str]] = None
) -> Optional[str]:











       
    if not text:
        return None

    filtered_text = text
    if markers:
        filtered_text = _strip_marker_blocks(filtered_text, markers)
    filtered_text = _TOOL_RESULT_BLOCK_PATTERN.sub("", filtered_text)
    filtered_text = _TOOL_CALL_BLOCK_PATTERN.sub("", filtered_text)

                                                                     
    boxed_matches = _BOXED_PATTERN.findall(filtered_text)
    if boxed_matches:
        for extracted in reversed(boxed_matches):
            if mode == 'choose':
                inner_matches = _TEXT_WRAPPER_PATTERN.findall(extracted)
                if inner_matches:
                    extracted = inner_matches[-1]
                extracted = extracted.strip("()")

            validated = _validate_answer(extracted)
            if validated:
                cleaned = _clean_answer(validated)
                if cleaned:
                    return cleaned

    extracted = _extract_boxed_answer(filtered_text)
    if extracted:
        if mode == 'choose':
            inner_matches = _TEXT_WRAPPER_PATTERN.findall(extracted)
            if inner_matches:
                extracted = inner_matches[-1]
            extracted = extracted.strip("()")

        validated = _validate_answer(extracted)
        if validated:
            cleaned = _clean_answer(validated)
            if cleaned:
                return cleaned

                                             
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(filtered_text)
        if match:
            extracted = match.group(1).strip()
                          
            extracted = extracted.split('\n')[0].strip()
                                             
            validated = _validate_answer(extracted)
            if validated and 3 <= len(validated) <= 500:
                return _clean_answer(validated)

                                          
    if mode == 'codegen':
        code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
        matches = code_pattern.findall(text)
        if matches:
            return matches[-1].strip()

    elif mode in ['infogen', 'summary', 'research']:
                                                            
        extracted = _extract_info_mode(text, mode)
        if extracted:
            return extracted

                                                             
    if mode == 'qa':
                     
        lines = filtered_text.strip().split('\n')
        skip_markers = ['let me', 'i will', 'i need', 'therefore,', 'so,', 'now,', 'first,', 'next,']

        for line in reversed(lines):
            line = line.strip()
            if not line or len(line) < 3 or len(line) > 500:
                continue
            line_lower = line.lower()
            if any(marker in line_lower for marker in skip_markers):
                continue
                        
            if '<tool_call>' in line or '</tool_call>' in line:
                continue
            validated = _validate_answer(line)
            if validated:
                return _clean_answer(validated)

    return None


def extract_answer_with_context(
    text: str,
    mode: str = 'qa',
    context_lines: int = 3,
    markers: Optional[Dict[str, str]] = None
) -> Tuple[Optional[str], Optional[str]]:












       
    if not text:
        return None, None

                                       
    answer = extract_answer(text, mode, markers=markers)
    if not answer:
        return None, None

                                           
                                                                    
    if '\\boxed{' in text:
        boxed_pos = text.rfind('\\boxed{')
                                                 
        context_start = max(0, boxed_pos - 500)
        context_end = min(len(text), boxed_pos + len(answer) + 100)
        context = text[context_start:context_end].strip()

                                              
        context_lines_list = [line for line in context.split('\n')
                             if '<tool_call>' not in line and '</tool_call>' not in line]
        context = '\n'.join(context_lines_list[-context_lines:]).strip()

        return answer, context

                                               
    lines = text.strip().split('\n')
                                           
    meaningful_lines = [line for line in lines
                       if line.strip()
                       and '<tool_call>' not in line
                       and '</tool_call>' not in line]

    if meaningful_lines:
        context = '\n'.join(meaningful_lines[-context_lines:]).strip()
        return answer, context

    return answer, answer


def _extract_boxed_answer(text: str) -> Optional[str]:
                                                                         
    if '\\boxed{' not in text:
        return None

    try:
                                                     
        matches = _BOXED_PATTERN.findall(text)
        if matches:
            return matches[-1].strip()

                                                                     
        start_idx = text.rfind('\\boxed{') + len('\\boxed{')
        brace_count = 1
        end_idx = start_idx

        while end_idx < len(text) and brace_count > 0:
            char = text[end_idx]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            end_idx += 1

        if brace_count == 0:
            return text[start_idx:end_idx-1].strip()

        return None
    except Exception:
        return None


def _validate_answer(answer: str) -> Optional[str]:









       
    if not answer:
        return None
    
                                
    if len(answer) > 500:
                                                  
                                                 
        number_match = re.search(r'\d+(?:\.\d+)?', answer[:200])
        if number_match:
            return number_match.group()
        else:
                                                          
            return answer[:100].strip()
    
                                                           
    if len(answer) > 10:
        char_counts = {}
        check_length = min(50, len(answer))
        for char in answer[:check_length]:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            max_repeat = max(char_counts.values())
                                                                                          
            if max_repeat > check_length * 0.7:
                return None                   
    
    return answer


def _extract_info_mode(text: str, mode: str) -> Optional[str]:










       
    extracted = ''
    
                            
    if "</think>" in text:
                                        
        extracted = text.split("</think>")[-1]
                                  
        extracted = extracted.split("<tool_call>")[0]
        extracted = extracted.split("<tool_call_result>")[0]
        extracted = extracted.strip()
        
        if mode == 'infogen':
                                     
            lines = extracted.replace("\n\n", "\n").split('\n')
            extracted = '\n'.join(lines[:5])
    else:
                                  
        lines = text.strip().replace("</think>", "").replace("\n\n", "\n").split('\n')
                                               
        meaningful_lines = [
            line for line in lines 
            if line.strip() 
            and '<tool_call>' not in line 
            and '</tool_call>' not in line
            and '<tool_call_result>' not in line
            and '</tool_call_result>' not in line
        ]
        
        if mode == 'infogen':
            extracted = '\n'.join(meaningful_lines[-5:])
        else:
            extracted = '\n'.join(meaningful_lines[-10:])
    
                         
    max_len = {'infogen': 2500, 'research': 6000, 'summary': 2500}.get(mode, 2500)
    extracted = extracted[:max_len]
    
    return _clean_answer(extracted) if extracted else None


def _clean_answer(answer: str) -> str:
                                                         
    if not answer:
        return answer

    answer = answer.strip()

                             
    for pattern, replacement in _CLEAN_PATTERNS:
        answer = pattern.sub(replacement, answer)

    answer = answer.strip()

                         
    if answer.endswith('.') and len(answer.split()) <= 5:
        answer = answer[:-1]

    return answer


def normalize_answer_qa(s: str) -> str:








       
    if not s:
        return ""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.strip().split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answer_math(text: str) -> str:








       
    if not text:
        return ""
    text = text.lower()
    text = " ".join(text.strip().split())
    return text


def count_tokens(text: str) -> int:








       
    global _tiktoken_encoder

    if not text:
        return 0

    try:
        import tiktoken
        if _tiktoken_encoder is None:
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        return len(_tiktoken_encoder.encode(text))
    except ImportError:
                                 
        char_count = len(text)
        word_count = len(text.split())
              
        char_estimate = char_count / 3.5
        word_estimate = word_count * 1.3
        return int(max(char_estimate, word_estimate))
    except Exception:
        return len(text.split())


def format_output(
    question: str,
    reasoning: str,
    answer: str,
    tool_calls: int = 0,
    folds: int = 0
) -> str:
                                         
    return f"""
{'='*80}
QUESTION: {question}
{'='*80}

REASONING:
{reasoning}

{'='*80}
FINAL ANSWER: {answer}
{'='*80}

STATISTICS:
- Tool Calls: {tool_calls}
- Memory Folds: {folds}
{'='*80}
"""


def clean_json_string(json_str: str) -> str:
                                                            
    if not json_str:
        return json_str

                                 
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)

    return json_str.strip()


def truncate_text(text: str, max_length: int = 1000) -> str:
                                                       
    if not text or len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def extract_json_content(text: str) -> str:
                                                                              
    if not text:
        return text

                                                    
    json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    match = json_pattern.search(text)
    if match:
        return match.group(1).strip()

                                      
    if '{' in text and '}' in text:
        start = text.find('{')
        end = text.rfind('}') + 1
        return text[start:end].strip()

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
                                                                    
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
                         
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def compute_text_similarity(text1: str, text2: str) -> float:
                                                                  
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0
