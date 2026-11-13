"""РОЛЬ A. Парный бутстрап по вопросам: доверительные интервалы режимов и их разности.

    from src.eval.bootstrap import paired_bootstrap, mean_ci
    r = paired_bootstrap(graph_by_qid, baseline_by_qid)    # graph минус baseline; любая пара режимов

Единица выборки — вопрос. Берутся только общие question_id двух режимов; вопрос, у которого в любом из режимов нет
оценки (None), исключается из ОБОИХ режимов (число исключённых — в результате и в сводке). Один и тот же набор
индексов вопросов (n_boot = 10000, seed = 42) используется для среднего каждого режима и для разности, поэтому
интервалы режимов и интервал разности согласованы. Интервал — перцентили 2,5 и 97,5 распределения бутстрап-средних.

Что показывает интервал: зависимость среднего от набора вопросов, а не разброс повторов. При температуре 0 повторы
ответов детерминированы; если в прогоне несколько повторов, на вопрос берётся их среднее, повторы вопросов не
размножаются.

Только numpy и стандартная библиотека.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

N_BOOT = 10_000
SEED = 42
PERCENTILES = (2.5, 97.5)
MIN_N = 5       # меньше вопросов — интервал не информативен

VERDICT_SMALL_N = "n<5, интервал не информативен"
VERDICT_NOT_PROVEN = "различие не доказано"
VERDICT_ZERO_EXCLUDED = "интервал не содержит 0"
VERDICT_NO_DATA = "нет общих вопросов с оценкой"
INTERVAL_NOTE = ("интервал показывает зависимость от набора вопросов, а не разброс повторов; "
                 "при температуре 0 повторы ответов детерминированы")


@dataclass
class Interval:
    mean: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n: int

    @property
    def note(self) -> Optional[str]:
        return VERDICT_SMALL_N if 0 < self.n < MIN_N else None


@dataclass
class Paired:
    """a_mode, b_mode — интервалы режимов на общих вопросах; diff — среднее (a − b) с интервалом."""
    a: Interval
    b: Interval
    diff: Interval
    n: int                  # общих вопросов с оценкой в обоих режимах (по ним всё посчитано)
    n_common: int           # общих вопросов всего
    n_excluded: int         # исключено из обоих режимов: у кого-то из режимов нет оценки
    verdict: str


def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _interval(x: np.ndarray, idx: np.ndarray) -> Interval:
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, PERCENTILES)
    return Interval(float(x.mean()), float(lo), float(hi), len(x))


def _empty() -> Interval:
    return Interval(None, None, None, 0)


def mean_ci(values: Mapping[str, Optional[float]], *, n_boot: int = N_BOOT, seed: int = SEED) -> Interval:
    """Среднее по вопросам с бутстрап-интервалом; вопросы без оценки (None) пропускаются."""
    xs = np.array([v for _, v in sorted(values.items()) if v is not None], dtype=float)
    if not len(xs):
        return _empty()
    return _interval(xs, _indices(len(xs), n_boot, seed))


def diff_verdict(diff: Interval) -> str:
    if diff.n == 0:
        return VERDICT_NO_DATA
    if diff.n < MIN_N:
        return VERDICT_SMALL_N
    if diff.ci_low <= 0 <= diff.ci_high:
        return VERDICT_NOT_PROVEN
    return VERDICT_ZERO_EXCLUDED


def paired_bootstrap(a: Mapping[str, Optional[float]], b: Mapping[str, Optional[float]], *,
                     n_boot: int = N_BOOT, seed: int = SEED) -> Paired:
    """a, b: {question_id: значение | None} двух режимов; разность считается как a − b."""
    common = sorted(set(a) & set(b))
    ok = [q for q in common if a[q] is not None and b[q] is not None]
    n_excl = len(common) - len(ok)
    if not ok:
        return Paired(_empty(), _empty(), _empty(), 0, len(common), n_excl, VERDICT_NO_DATA)
    xa = np.array([a[q] for q in ok], dtype=float)
    xb = np.array([b[q] for q in ok], dtype=float)
    idx = _indices(len(ok), n_boot, seed)    # один набор индексов на оба режима и разность: пары не рвутся
    diff = _interval(xa - xb, idx)
    return Paired(_interval(xa, idx), _interval(xb, idx), diff, len(ok), len(common), n_excl, diff_verdict(diff))
