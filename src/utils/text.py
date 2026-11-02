"""Единая нормализация текста для сравнения цитат и имён (ru/kk/en).

NFC нужен, чтобы казахские буквы (ә ғ қ ң ө ұ ү һ і) и «й/ё» совпадали,
даже если ASR отдал их составными символами.
"""
import re
import unicodedata


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()
