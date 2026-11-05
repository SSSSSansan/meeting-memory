"""B4. Разрешение сущностей: «Атлас» / «проект Атлас» / «Жол» / «Zhol» → один узел.

Без LLM: нормализация + транслитерация в латиницу + нечёткое совпадение. Спорное (ниже порога)
не склеивается, а остаётся отдельным узлом — ложная склейка хуже пропущенной.
    resolve_projects(names, known=None) / resolve_people(names, known=None) → {как_прозвучало: каноническое}
`known` — канонические имена (участники встречи, проекты из реестра); без них канон = самый частый вариант.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "і": "i", "й": "i", "к": "k", "қ": "k", "л": "l", "м": "m", "н": "n", "ң": "n", "о": "o",
    "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f", "х": "h",
    "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ә": "a", "ю": "iu", "я": "ia",
}
_PREFIXES = ("проект ", "project ", "жоба ", "приложение ")
_DIMINUTIVES = {"коля": "николай", "николя": "николай", "саша": "александр", "женя": "евгений"}
THRESHOLD = 0.8


def key(name: str) -> str:
    """Ключ сравнения: нижний регистр, без «проект», кириллица → латиница, похожие звуки схлопнуты."""
    s = name.lower().strip()
    for p in _PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    s = _DIMINUTIVES.get(s, s)
    k = "".join(_CYR.get(ch, ch) for ch in s if ch.isalnum() or ch == " ").replace(" ", "")
    for a, b in (("ph", "f"), ("kh", "h"), ("ck", "k"), ("q", "k"), ("w", "u"), ("v", "u"), ("j", "zh"), ("dzh", "zh"),
                 ("y", "i"), ("ii", "i"), ("oo", "u"), ("ou", "u"), ("o", "a"), ("e", "a")):
        k = k.replace(a, b)
    return k


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 3 and long_.startswith(short):  # Абду → Абдурахман
        return True
    return SequenceMatcher(None, a, b).ratio() > THRESHOLD


def _resolve(names: list[str], known: list[str] | None) -> dict[str, str]:
    freq = Counter(names)
    canon: list[str] = list(known or [])
    out: dict[str, str] = {}
    for name, _ in freq.most_common():  # самые частые первыми — они задают канон
        k = key(name)
        hit = next((c for c in canon if _same(k, key(c))), None)
        if hit is None:
            canon.append(name)
            hit = name
        out[name] = hit
    return out


def resolve_projects(names: list[str], known: list[str] | None = None) -> dict[str, str]:
    return _resolve(names, known)


def resolve_people(names: list[str], known: list[str] | None = None) -> dict[str, str]:
    return _resolve(names, known)
