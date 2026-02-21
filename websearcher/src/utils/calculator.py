import ast
import math
from typing import Any, Dict, Optional


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)
_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp
}
_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e
}


class CalculatorError(Exception):
    pass


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError("Only numeric constants are allowed.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        operand = _eval_node(node.operand)
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only simple function calls are allowed.")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCS:
            raise CalculatorError(f"Function '{func_name}' is not allowed.")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCS[func_name](*args)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise CalculatorError(f"Name '{node.id}' is not allowed.")

    raise CalculatorError(f"Unsupported expression: {type(node).__name__}")


def evaluate_expression(expression: str, precision: Optional[int] = None) -> Dict[str, Any]:
    if not expression or not expression.strip():
        raise CalculatorError("Empty expression.")

    sanitized = expression.strip().replace("^", "**")
    try:
        tree = ast.parse(sanitized, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {exc}") from exc

    result = _eval_node(tree)
    if isinstance(precision, int) and precision >= 0:
        result = round(result, precision)

    return {
        "expression": expression,
        "result": result
    }
