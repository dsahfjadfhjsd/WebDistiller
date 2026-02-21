import ast
import io
import tokenize
from pathlib import Path

ROOT = Path('.')
py_files = [p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts]


def remove_backslash_only_lines(lines):
    new_lines = []
    for line in lines:
        if line.strip() == "\\":
            
            new_lines.append("\n" if line.endswith("\n") else "")
        else:
            new_lines.append(line)
    return new_lines


def get_docstring_ranges(source):
    ranges = []
    try:
        tree = ast.parse(source)
    except Exception:
        return ranges

    def visit(node):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr):
                    value = first.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        start = first.lineno - 1
                        end = (first.end_lineno - 1) if hasattr(first, 'end_lineno') and first.end_lineno else start
                        ranges.append((start, end))
                    elif isinstance(value, ast.Str):
                        start = first.lineno - 1
                        end = (first.end_lineno - 1) if hasattr(first, 'end_lineno') and first.end_lineno else start
                        ranges.append((start, end))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return ranges


def strip_comments_and_docstrings(source):
    lines = source.splitlines(True)
    lines = remove_backslash_only_lines(lines)
    source = ''.join(lines)

    
    for start, end in get_docstring_ranges(source):
        for i in range(start, end + 1):
            if i < len(lines):
                lines[i] = "\n" if lines[i].endswith("\n") else ""

    source = ''.join(lines)

    
    comment_positions = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                line_no, col = tok.start
                end_col = tok.end[1]
                comment_positions.setdefault(line_no - 1, []).append((col, end_col))
    except Exception:
        comment_positions = {}

    if comment_positions:
        lines = source.splitlines(True)
        for line_no, spans in comment_positions.items():
            if line_no >= len(lines):
                continue
            line = lines[line_no]
            
            for start_col, end_col in sorted(spans, reverse=True):
                lines[line_no] = line[:start_col] + line[end_col:]
                line = lines[line_no]
        source = ''.join(lines)

    
    lines = source.splitlines(True)
    lines = remove_backslash_only_lines(lines)
    return ''.join(lines)


changed = 0
for path in py_files:
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='latin-1')
    new_text = strip_comments_and_docstrings(text)
    if new_text != text:
        path.write_text(new_text, encoding='utf-8')
        changed += 1

print(f"Processed {len(py_files)} files, updated {changed} files")
