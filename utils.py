from typing import Any
from moderngl import Program

def safe_set_uniform(prog:Program, name: str, value: Any):
    if name in prog:
        prog[name].value = value
