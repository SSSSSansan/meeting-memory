"""Сводка итоговых таблиц строится из сохранённых json без модели."""
from scripts import final_report_tables as t


def test_report_builds_and_has_all_hypothesis_items():
    text = t.build()
    assert "## 5. Проверка гипотезы" in text
    assert text.count("- Вердикт:") == 7
    assert "условие не наступило" in text


def test_group_split():
    assert t.in_group("T1", "T1") and not t.in_group("T2", "T1")
    assert t.in_group("T5", "T2-T5") and not t.in_group("none", "T2-T5")
