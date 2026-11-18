"""Small AST interpreter for rollout calculator tools; never eval/execs model text."""

import ast
import operator

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_calculator(expression):
    if len(expression) > 512:
        return None
    expression = expression.replace(",", "") if all(c in "0123456789,*+-/.() " for c in expression) else expression
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return None

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
                return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
                value = visit(node.operand)
                if type(value) in (int, float):
                    return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                left, right = visit(node.left), visit(node.right)
                if type(left) in (int, float) and type(right) in (int, float):
                    result = _BINARY[type(node.op)](left, right)
                    if abs(result) <= 1e100:
                        return result
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "count" and not node.keywords and len(node.args) == 1):
                text, substring = visit(node.func.value), visit(node.args[0])
                if isinstance(text, str) and isinstance(substring, str):
                    return text.count(substring)
            raise ValueError("Unsupported calculator expression")

        return visit(tree.body)
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return None