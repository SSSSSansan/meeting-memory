"""РОЛЬ A. Eval-harness: прогон обеих веток по data/gold/questions.json.

TODO (день 4):
    python -m src.eval.run --mode both --repeats 3
  - метрики из раздела 8 контекста (F1 извлечения, hallucination rate, accuracy,
    citation correctness, токены, $, задержка)
  - 3 повтора, среднее ± разброс
  - итог: results/summary.csv, главная таблица baseline vs graph по типам T1–T5
"""


def main() -> None:
    raise NotImplementedError("Роль A")


if __name__ == "__main__":
    main()
