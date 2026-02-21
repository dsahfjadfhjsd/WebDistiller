import os
import io
import json
import regex
import pickle
import traceback
import copy
import datetime
try:
    import dateutil.relativedelta
except Exception:
    dateutil = None
import asyncio
import sys
import tempfile
from typing import Any, Dict, Optional
from contextlib import redirect_stdout
try:
    import numpy as np
except Exception:
    np = None

try:
    import sympy
except Exception:
    sympy = None
import math
import random
import statistics
import itertools
import collections
try:
    from sympy import symbols, Eq, solve
except Exception:
    symbols = None
    Eq = None
    solve = None


class UnsafeCodeError(Exception):
    pass

_PYTHON_ERROR_LOG_PATH: Optional[str] = None


def _get_python_error_log_path() -> str:
    global _PYTHON_ERROR_LOG_PATH
    if _PYTHON_ERROR_LOG_PATH:
        return _PYTHON_ERROR_LOG_PATH
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_dir = os.path.join(repo_root, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = os.getcwd()
    _PYTHON_ERROR_LOG_PATH = os.path.join(log_dir, "python_executor_errors.log")
    return _PYTHON_ERROR_LOG_PATH


def _log_python_error(
    stage: str,
    message: str,
    code: str = "",
    question: str = "",
    stderr: str = "",
    extra: Optional[Dict[str, Any]] = None
) -> None:
    try:
        log_path = _get_python_error_log_path()
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        code_snippet = code if len(code) <= 4000 else code[:4000] + "\n...[truncated]..."
        stderr_snippet = stderr if len(stderr) <= 4000 else stderr[:4000] + "\n...[truncated]..."
        payload = {
            "time": timestamp,
            "stage": stage,
            "message": message,
            "question": question or "",
            "code": code_snippet,
            "stderr": stderr_snippet
        }
        if extra:
            payload["extra"] = extra
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

UNSAFE_PATTERNS = [
    # Note: os is allowed in sandbox header, but we still block it here to prevent misuse
    # Users should use os.path.exists(), os.listdir() etc. which are provided by the sandbox
    r'import\s+(sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread|requests|urllib|pandas)',
    r'from\s+(sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread|requests|urllib|pandas)\s+import',
    # Block network libraries (more specific patterns)
    r'import\s+requests',
    r'from\s+requests\s+import',
    r'import\s+urllib',
    r'from\s+urllib\s+import',
    # Block data manipulation libraries
    r'import\s+pandas',
    r'from\s+pandas\s+import',
    # Block dangerous os operations even if os is imported
    r'os\.(system|popen|fork|kill|remove|rmdir|chmod|chown)',
    # Block dangerous built-in functions
    r'(?<!\w)(input|eval|exec|exit|quit|__import__)\s*\(',
]


def validate_python_code(code: str) -> tuple:




    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Code validation error: {str(e)}"


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

                                                          
        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
                                                 
        for pattern in UNSAFE_PATTERNS:
            if regex.search(pattern, code_piece):
                raise UnsafeCodeError("Your process is not safe. Execution of potentially unsafe code was blocked.")

                            
        imports = """
import numpy as np
import sympy
import math
import random
import statistics
import itertools
import collections
from collections import Counter, defaultdict
try:
    from sympy import symbols, Eq, solve, simplify, expand, factor
    x, y, z, a, b, c, n, m = sympy.symbols('x y z a b c n m') if sympy else (None,) * 8
except Exception:
    symbols = None
    Eq = solve = simplify = expand = factor = None
    x = y = z = a = b = c = n = m = None

from fractions import Fraction
from decimal import Decimal
"""
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()

                                         
        exec(imports, self._global_vars)
                                
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

_RELATIVEDELTA = dateutil.relativedelta.relativedelta if dateutil else datetime.timedelta


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime, 
        'timedelta': _RELATIVEDELTA,
        'relativedelta': _RELATIVEDELTA
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 5,
        max_concurrency: int = 8,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    @staticmethod
    async def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        runtime = runtime if runtime else GenericRuntime()
        get_answer_from_stdout = get_answer_from_stdout if get_answer_from_stdout is not None else False
        try:
                                                        
            if isinstance(code, list):
                code = '\n'.join(code)
            
                                           
            code = code.strip()

            def run_exec_sync(code_snippet: str):
                return runtime.exec_code(code_snippet)

            async def run_eval(expr: str):
                return await asyncio.to_thread(runtime.eval_code, expr)
            
            if get_answer_from_stdout:
                program_io = io.StringIO()
                def _exec_with_capture():
                    with redirect_stdout(program_io):
                        runtime.exec_code(code)
                await asyncio.wait_for(asyncio.to_thread(_exec_with_capture), timeout=timeout_length)
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                await asyncio.wait_for(run_exec_sync(code), timeout=timeout_length)
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                await asyncio.wait_for(run_exec_sync(code), timeout=timeout_length)
                result = await asyncio.wait_for(run_eval(answer_expr), timeout=timeout_length)
            else:
                                                         
                code_lines = code.split('\n')
                if len(code_lines) > 1:
                    exec_code = '\n'.join(code_lines[:-1])
                    eval_code = code_lines[-1]
                    await asyncio.wait_for(run_exec_sync(exec_code), timeout=timeout_length)
                    result = await asyncio.wait_for(run_eval(eval_code), timeout=timeout_length)
                else:
                    result = await asyncio.wait_for(run_eval(code), timeout=timeout_length)
                    
            report = "Done"
            
                                      
            try:
                                  
                pickle.dumps(result)
            except (pickle.PicklingError, TypeError):
                                                               
                try:
                    result = str(result)
                except Exception:
                                                                              
                    result = f"<unprintable object of type {type(result).__name__}>"
            
        except Exception as e:
            result = ''
            report = str(e)
        return result, report

    @staticmethod
    def execute_sync(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        return asyncio.run(PythonExecutor.execute(
            code,
            get_answer_from_stdout=get_answer_from_stdout,
            runtime=runtime,
            answer_symbol=answer_symbol,
            answer_expr=answer_expr,
            timeout_length=timeout_length,
        ))

    async def apply(self, code):
        return (await self.batch_apply([code]))[0]

    def apply_sync(self, code):
        return asyncio.run(self.apply(code))

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    async def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        async def _bounded_execute(snippet):
            async with self._semaphore:
                return await PythonExecutor.execute(
                    snippet,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime=self.runtime,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    timeout_length=self.timeout_length,
                )

        tasks = [asyncio.create_task(_bounded_execute(code_snippet)) for code_snippet in all_code_snippets]
        all_exec_results = await asyncio.gather(*tasks, return_exceptions=False)

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
                             
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results

    def batch_apply_sync(self, batch_code):
        return asyncio.run(self.batch_apply(batch_code))

async def execute_python_code(
    code: str,
    timeout: int = 30,
    question: str = "",
    base_dir: Optional[str] = None
) -> str:
















    import json
    import importlib.util
    import re
    
             
    try:
        from .code_validator import validate_python_result
        validator_available = True
    except ImportError:
        validator_available = False

                                             
    for pattern in UNSAFE_PATTERNS:
        if regex.search(pattern, code):
            # 提供具体的错误提示
            hint = "Unsafe code pattern detected. Network access (requests, urllib) and file system operations (os, sys) are restricted in the sandbox."
            
            # 根据检测到的模式提供更具体的提示
            if regex.search(r'import\s+requests|from\s+requests', code) or regex.search(r'import\s+urllib|from\s+urllib', code):
                hint = (
                    "Network access is disabled in the Python sandbox. "
                    "The 'requests' and 'urllib' modules cannot be used. "
                    "If you need to fetch data from the web, use the 'web_search' or 'download_file' tools instead."
                )
            elif regex.search(r'import\s+pandas|from\s+pandas', code):
                hint = (
                    "The 'pandas' library is not available in the sandbox environment. "
                    "For data manipulation, consider using built-in Python data structures (lists, dicts) "
                    "or the available libraries: math, random, statistics, itertools, collections."
                )
            elif regex.search(r'import\s+os|from\s+os', code):
                hint = (
                    "The 'os' module import is restricted. However, os.path.exists(), os.listdir(), and other file operations "
                    "are available in the sandbox without importing os. You can use them directly (e.g., os.path.exists('file.txt')). "
                    "For file operations, use the 'download_file' and 'process_file' tools instead."
                )
            
            _log_python_error(
                stage="unsafe_code",
                message="Unsafe code pattern detected. Execution blocked for security reasons.",
                code=code,
                question=question,
                extra={"pattern": pattern}
            )
            return json.dumps({
                "error": "Unsafe code pattern detected. Execution blocked for security reasons.",
                "status": "unsafe_code",
                "hint": hint
            })

             
    is_valid, validation_error = validate_python_code(code)
    if not is_valid:
        _log_python_error(
            stage="syntax_error",
            message=validation_error,
            code=code,
            question=question
        )
        return json.dumps({
            "error": validation_error,
            "status": "syntax_error",
            "hint": "Please check your code syntax and try again."
        })

    def _missing_dep_message(dep_name: str) -> str:
        return (
            f"Python dependency '{dep_name}' is not available in the current environment. "
            f"Install it and retry (e.g., `pip install {dep_name}` or "
            f"`pip install -r requirements.txt`)."
        )

    numpy_needed = bool(re.search(r'(^|\s)(import|from)\s+numpy\b|np\.', code))
    sympy_needed = bool(re.search(r'(^|\s)(import|from)\s+sympy\b|sympy\.', code))

    if numpy_needed and importlib.util.find_spec("numpy") is None:
        _log_python_error(
            stage="missing_dependency",
            message=_missing_dep_message("numpy"),
            code=code,
            question=question,
            extra={"dependency": "numpy"}
        )
        return json.dumps({
            "error": _missing_dep_message("numpy"),
            "status": "missing_dependency"
        })
    if sympy_needed and importlib.util.find_spec("sympy") is None:
        _log_python_error(
            stage="missing_dependency",
            message=_missing_dep_message("sympy"),
            code=code,
            question=question,
            extra={"dependency": "sympy"}
        )
        return json.dumps({
            "error": _missing_dep_message("sympy"),
            "status": "missing_dependency"
        })

    try:
        # NOTE: The previous implementation executed code inside a background thread and
        # used asyncio.wait_for for timeout. In CPython, cancelling run_in_executor/to_thread
        # does NOT stop the underlying thread. For CPU-heavy / infinite-loop code, this can
        # cause the tool call to hang indefinitely. We execute in a separate process and
        # hard-kill on timeout to guarantee a response.

        resolved_base_dir = os.path.abspath(base_dir) if base_dir else os.getcwd()
        if not os.path.isdir(resolved_base_dir):
            resolved_base_dir = os.getcwd()
        SANDBOX_HEADER = f"""\
# Sandbox environment setup
# Note: os, builtins, socket are already imported. You can use os.path.exists(), os.listdir(), etc. without importing os.
import os
import builtins
import socket

_BASE_DIR = {repr(resolved_base_dir)}
DOWNLOAD_DIR = "downloads"  # Files downloaded via download_file are stored in this directory (relative to base_dir)

try:
    import numpy as np
except Exception:
    np = None

try:
    import sympy
except Exception:
    sympy = None

import math
import random
import statistics
import itertools
import collections
from collections import Counter, defaultdict
try:
    from sympy import symbols, Eq, solve, simplify, expand, factor
    x, y, z, a, b, c, n, m = sympy.symbols('x y z a b c n m') if sympy else (None,) * 8
except Exception:
    symbols = None
    Eq = solve = simplify = expand = factor = None
    x = y = z = a = b = c = n = m = None

from fractions import Fraction
from decimal import Decimal

def _safe_path(path):
    if os.path.isabs(path):
        candidate = os.path.abspath(path)
    else:
        candidate = os.path.abspath(os.path.join(_BASE_DIR, path))
    try:
        if os.path.commonpath([_BASE_DIR, candidate]) != _BASE_DIR:
            raise PermissionError("File access outside sandbox is not allowed.")
    except Exception:
        raise PermissionError("File access outside sandbox is not allowed.")
    return candidate

_real_open = builtins.open
def _safe_open(file, *args, **kwargs):
    return _real_open(_safe_path(file), *args, **kwargs)
builtins.open = _safe_open
os.chdir(_BASE_DIR)

_real_listdir = os.listdir
def _safe_listdir(path="."):
    return _real_listdir(_safe_path(path))
os.listdir = _safe_listdir

_real_scandir = os.scandir
def _safe_scandir(path="."):
    return _real_scandir(_safe_path(path))
os.scandir = _safe_scandir

_real_walk = os.walk
def _safe_walk(top, *args, **kwargs):
    return _real_walk(_safe_path(top), *args, **kwargs)
os.walk = _safe_walk

_real_exists = os.path.exists
def _safe_exists(path):
    try:
        return _real_exists(_safe_path(path))
    except Exception:
        return False
os.path.exists = _safe_exists

_real_isfile = os.path.isfile
def _safe_isfile(path):
    try:
        return _real_isfile(_safe_path(path))
    except Exception:
        return False
os.path.isfile = _safe_isfile

_real_isdir = os.path.isdir
def _safe_isdir(path):
    try:
        return _real_isdir(_safe_path(path))
    except Exception:
        return False
os.path.isdir = _safe_isdir

def _blocked_socket(*args, **kwargs):
    raise PermissionError("Network access disabled in Python tool.")
socket.socket = _blocked_socket
"""

        script = f"{SANDBOX_HEADER}\n{code}\n"

        async def _read_stream_limited(stream: asyncio.StreamReader, limit_bytes: int) -> tuple[str, bool]:
            chunks: list[bytes] = []
            total = 0
            truncated = False
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total <= limit_bytes:
                    chunks.append(chunk)
                else:
                    truncated = True
                    # Keep draining but only keep up to the limit in memory
                    prev_total = total - len(chunk)
                    if prev_total < limit_bytes:
                        chunks.append(chunk[: (limit_bytes - prev_total)])
            data = b"".join(chunks)
            return data.decode("utf-8", errors="replace"), truncated

        MAX_STDOUT_BYTES = 200_000
        MAX_STDERR_BYTES = 200_000

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as f:
                f.write(script)
                tmp_path = f.name

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_base_dir
            )

            assert proc.stdout is not None
            assert proc.stderr is not None

            stdout_task = asyncio.create_task(_read_stream_limited(proc.stdout, MAX_STDOUT_BYTES))
            stderr_task = asyncio.create_task(_read_stream_limited(proc.stderr, MAX_STDERR_BYTES))

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                # Ensure the subprocess transport is cleaned up on Windows:
                # wait for termination and drain pipes (otherwise __del__ may log WinError 6).
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task, return_exceptions=True), timeout=3)
                except Exception:
                    pass
                _log_python_error(
                    stage="timeout",
                    message=f"Code execution timeout ({timeout}s).",
                    code=code,
                    question=question
                )
                return json.dumps({
                    "error": f"Code execution timeout ({timeout}s). Consider optimizing your code or using simpler algorithms.",
                    "status": "timeout",
                    "hint": "Try reducing search space / adding pruning. For large loops, prefer vectorized numpy operations."
                })

            stdout_text, stdout_trunc = await stdout_task
            stderr_text, stderr_trunc = await stderr_task

            if proc.returncode == 0:
                result_str = stdout_text
                if stdout_trunc:
                    result_str = result_str + "\n[stdout truncated]"

                validation_warning = None
                if validator_available:
                    validation = validate_python_result(code, result_str, question)
                    if not validation["is_valid"]:
                        validation_warning = validation["warning"]
                        if validation["suggestions"]:
                            validation_warning += "\n\n" + validation["suggestions"]

                response = {
                    "result": result_str,
                    "status": "success"
                }
                if stderr_text.strip():
                    response["stderr"] = (stderr_text + ("\n[stderr truncated]" if stderr_trunc else "")).strip()

                if validation_warning:
                    response["validation_warning"] = validation_warning
                    response["status"] = "success_with_warnings"

                return json.dumps(response)

            # Non-zero return code: treat as failed and provide stderr as the error message.
            error_text = (stderr_text or "").strip() or f"Process exited with code {proc.returncode}"
            if stderr_trunc:
                error_text += "\n[stderr truncated]"
            hint = "Check your code logic and try again."
            
            # 提供更详细的错误提示（保持泛化性）
            if "IndexError: list index out of range" in error_text:
                hint = (
                    "Index out of range. Check list/array length before indexing. "
                    "Consider adding bounds checking (e.g., `if len(list) > index`)."
                )
            elif "KeyError" in error_text:
                hint = "Key not found in dictionary. Add existence checks before dict access (e.g., `if key in dict`)."
            elif "ZeroDivisionError" in error_text:
                hint = "Division by zero. Add guards for zero denominators before division operations."
            elif "AttributeError" in error_text:
                hint = "Attribute not found. Check if the object has the expected attribute before accessing it."
            elif "TypeError" in error_text:
                hint = "Type error. Check variable types and ensure operations are compatible with the data types used."
            elif "NameError" in error_text:
                hint = "Variable not defined. Check if all variables are properly initialized before use."
            elif "ImportError" in error_text or "ModuleNotFoundError" in error_text:
                hint = (
                    "Module import failed. The required library may not be available in the sandbox environment. "
                    "Common available libraries: math, random, statistics, itertools, collections, fractions, decimal. "
                    "For data manipulation, consider using built-in Python data structures instead of pandas/numpy."
                )
            elif "ValueError" in error_text:
                hint = "Invalid value. Check input values and ensure they meet the function's requirements."
            elif "FileNotFoundError" in error_text:
                hint = "File not found. Check the file path and ensure the file exists in the allowed directory."
            _log_python_error(
                stage="execution_failed",
                message=error_text,
                code=code,
                question=question,
                stderr=stderr_text,
                extra={"returncode": proc.returncode}
            )

            return json.dumps({
                "error": error_text,
                "status": "failed",
                "hint": hint
            })
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    except asyncio.TimeoutError:
        _log_python_error(
            stage="timeout",
            message=f"Code execution timeout ({timeout}s).",
            code=code,
            question=question
        )
        return json.dumps({
            "error": f"Code execution timeout ({timeout}s). Consider optimizing your code or using simpler algorithms.",
            "hint": "For simulations, consider reducing iterations or using vectorized operations with numpy."
        })
    except UnsafeCodeError as e:
        error_msg = str(e)
        hint = "Unsafe code pattern detected. Network access (requests, urllib) and file system operations (os, sys) are restricted in the sandbox."
        
        # 提供更具体的提示
        if "import" in code.lower() and ("requests" in code.lower() or "urllib" in code.lower()):
            hint = (
                "Network access is disabled in the Python sandbox. "
                "The 'requests' and 'urllib' modules cannot be used. "
                "If you need to fetch data from the web, use the 'web_search' or 'download_file' tools instead."
            )
        elif "import os" in code.lower() or "from os" in code.lower():
            hint = (
                "The 'os' module import is restricted. However, os.path.exists(), os.listdir(), and other file operations "
                "are available in the sandbox without importing os. You can use them directly (e.g., os.path.exists('file.txt')). "
                "For file operations, use the 'download_file' and 'process_file' tools instead."
            )
        
        _log_python_error(
            stage="unsafe_code",
            message=error_msg,
            code=code,
            question=question,
            extra={"pattern": "unsafe_code"}
        )
        return json.dumps({
            "error": error_msg,
            "status": "unsafe_code",
            "hint": hint
        })
    except Exception as e:
        _log_python_error(
            stage="exception",
            message=str(e),
            code=code,
            question=question,
            stderr=traceback.format_exc()
        )
        return json.dumps({
            "error": f"Execution error: {str(e)}",
            "traceback": traceback.format_exc()[-500:]                               
        })
    

def execute_python_code_sync(code: str) -> str:









    try:
        executor = PythonExecutor(get_answer_from_stdout=True)
        result, report = executor.apply_sync(code)
        returned_result = "Execution result: " + result + "\nExecution status: " + report
        return returned_result
    except Exception as e:
        return f"Execution error: {str(e)}"
    

def get_openai_function_execute_python_code(file_process: bool = False):
    if file_process:
        return {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code in a safe sandbox environment and return the execution results from stdout. This could help you with mathematical computations, reading tables, data analysis, and general computation-intensive tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                            "code": {
                            "type": "string",
                            "description": "The Python code to execute. Note: all files are located in '/mnt/tidalfs-bdsz01/usr/tusen/xiaoxi/DeepAgent/data/GAIA/files/'. Please use the absolute path when accessing any files.",
                            "examples": [
                                "x = 5\nprint(x * 2)",
                                "import sympy\nx = sympy.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sympy.factor(expr))",
                                "import pandas as pd\ndf = pd.read_csv('/mnt/tidalfs-bdsz01/usr/tusen/xiaoxi/DeepAgent/data/GAIA/files/example.csv')\nprint(df.head())"
                            ]
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code in a safe sandbox environment and return the execution results from stdout. This could help you with mathematical computations, reading tables, data analysis, and general computation-intensive tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute. Avoid using unsafe functions or packages.",
                            "examples": [
                                "x = 5\nprint(x * 2)",
                                "import sympy\nx = sympy.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sympy.factor(expr))",
                                "import pandas as pd\ndf = pd.read_csv('example.csv')\nprint(df.head())"
                            ]
                        }
                    },
                    "required": ["code"]
                }
            }
        }


async def _test():
    code = "import sympy\nx = sympy.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sympy.factor(expr))"
    result = await execute_python_code(code)
    print(result)


if __name__ == '__main__':
    asyncio.run(_test())

