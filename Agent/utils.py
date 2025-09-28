import re
import textwrap

_PY_FENCE = re.compile(
    r"```(?:python|py|python3)(?:[^\n]*)\r?\n(.*?)(?:\r?\n)?```",
    flags=re.IGNORECASE | re.DOTALL,
)

_ANY_FENCE = re.compile(
    r"```\s*\r?\n(.*?)(?:\r?\n)?```",
    flags=re.DOTALL,
)

def extract_python_code(text: str) -> str:
    """
    Extract the first ```python ... ``` fenced block from LLM output.
    Falls back to the first unlabeled ``` ... ``` block.
    If neither is found, return the original text (fallback-to-raw).
    """
    m = _PY_FENCE.search(text) or _ANY_FENCE.search(text)
    code = m.group(1) if m else text
    return textwrap.dedent(code).strip("\n")

if __name__ == "__main__":
    demo = """
    Here’s some explanation.

    ```python
    import math

    def area(r: float) -> float:
        return math.pi * r * r
    ```

    and that's the end of the code.
    """
    print(extract_python_code(demo))
    demo = """
    import math

    def area(r: float) -> float:
        return math.pi * r * r
        """
    print(extract_python_code(demo))

