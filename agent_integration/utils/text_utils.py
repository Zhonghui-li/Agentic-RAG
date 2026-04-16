# utils/text_utils.py
from __future__ import annotations
from typing import List

# ========== 基础分词器（粗略）：按空白切分 ==========
def _whitespace_tokenize(text: str) -> List[str]:
    return text.split()

# ========== 主函数：按 token 数上限截断 ==========
def trim_text_to_token_limit(
    text: str,
    max_tokens: int,
    model: str | None = None,
    **kwargs,
) -> str:
    """
    将文本按空白粗略分词，并在不超过 max_tokens 的情况下返回拼接结果。
    - model/kwargs 仅为兼容旧调用，不参与逻辑。
    """
    if not text:
        return text
    if max_tokens is None or max_tokens <= 0:
        return ""
    tokens = _whitespace_tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])

# ========== 兼容别名：旧代码里可能调用了这个名字 ==========
def trim_text_to_tokens(
    text: str,
    max_tokens: int,
    model: str | None = None,
    **kwargs,
) -> str:
    """
    兼容别名。等价于 trim_text_to_token_limit。
    """
    return trim_text_to_token_limit(text, max_tokens=max_tokens, model=model, **kwargs)

# ========== 安全截断长 prompt：保留头尾，裁中间 ==========
def safe_trim_prompt(
    prompt: str,
    max_tokens: int = 2048,
    model: str | None = None,
    **kwargs,
) -> str:
    """
    安全截断：保留开头的指令和结尾的用户输入；中间过长时裁掉。
    - 先按行切分保留结构，再对白名单中间块做基于 token 的粗截断。
    - model/kwargs 仅为兼容旧调用。
    """
    if not prompt:
        return prompt
    if max_tokens is None or max_tokens <= 0:
        return ""

    # 经验值：按行保留两端结构
    lines = prompt.splitlines()
    # 先快速检查：若整体 token 已经不超，就直接返回
    if len(_whitespace_tokenize(prompt)) <= max_tokens:
        return prompt

    # 头尾行数可按需调整
    keep_head = 20
    keep_tail = 20

    if len(lines) <= keep_head + keep_tail:
        # 行数不多，直接基于 token 粗截断
        return trim_text_to_token_limit(prompt, max_tokens=max_tokens)

    head = lines[:keep_head]
    tail = lines[-keep_tail:]
    middle = lines[keep_head: len(lines) - keep_tail]

    # 给中间部分剩余 token 配额（至少留一些 token）
    head_tokens = len(_whitespace_tokenize("\n".join(head)))
    tail_tokens = len(_whitespace_tokenize("\n".join(tail)))
    remaining = max(max_tokens - head_tokens - tail_tokens, 16)

    middle_text = "\n".join(middle)
    middle_trimmed = trim_text_to_token_limit(middle_text, max_tokens=remaining)

    return "\n".join(head + [middle_trimmed] + tail)
