from dataclasses import dataclass, replace
from math import fsum
from typing import Any

from .baseline import cohort_pick_times, pickup_estimate
from .huber_model import score_kernel

MIN_PAIRED_ORDERS = 6
MIN_PAIRED_SAMPLES = 6
MIN_REPLAY_ORDERS = 6
REQUIRED_PAIRED_ORDERS = 6
REGRESSION = 'regression'
MODEL = REGRESSION
BANDS = 'bands'
BAND = BANDS
BASELINE = 'baseline'
REGRESSION_LABEL = 'Huber 稳健时长回归'
BANDS_LABEL = '人均行数分档中位'
REGRESSION_OPTION = '回归估时'
BANDS_OPTION = '分档估时'

REASON_INSUFFICIENT = 'INSUFFICIENT_PAIRED_ORDERS'
REASON_MAE = 'MAE_NOT_LOWER'
REASON_LATE_LOAD = 'LATE_LOAD_MISS_OVERRIDE'
REASON_WINS = 'REGRESSION_REPLAY_WINS'
REASON_NO_REPLAY = 'NO_PAIRED_REPLAY'

_REASON_TEXT = {
    REASON_INSUFFICIENT: '同区成对实绩不足六单，使用分档估时',
    REASON_MAE: '回归平均绝对误差未严格更小，使用分档估时',
    REASON_LATE_LOAD: '回归多漏查晚装车单，安全闸门保留分档估时',
    REASON_WINS: '回归误差更低且未多漏查晚装车单，启用回归估时',
    REASON_NO_REPLAY: '本区没有可成对比较的回放实绩，使用分档估时',
}


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    key: str
    zone: str
    regression: float
    bands: float
    actual: float
    buffer_minutes: float
    buffer_known: bool = True

    @property
    def regression_absolute_error(self) -> float:
        return abs(self.regression - self.actual)

    @property
    def bands_absolute_error(self) -> float:
        return abs(self.bands - self.actual)

    def misses_late_load(self, estimate: float) -> bool | None:
        """Miss a ticket-specific late load under the dispatch boundary.

        HANDOFF_AT_RISK is raised only when ``estimate > slack_min - 6``.
        Its negation is therefore ``estimate <= buffer``; the ticket is late
        when ``actual > buffer``.
        """
        if not self.buffer_known:
            return None
        return estimate <= self.buffer_minutes < self.actual

    def misses_diagnostic_boundary(self, estimate: float) -> bool | None:
        """Compatibility form matching ``estimate < buffer <= actual``."""
        if not self.buffer_known:
            return None
        return estimate < self.buffer_minutes <= self.actual

    @property
    def regression_missed(self) -> bool | None:
        return self.misses_late_load(self.regression)

    @property
    def bands_missed(self) -> bool | None:
        return self.misses_late_load(self.bands)


@dataclass(frozen=True, slots=True)
class ZoneChoice:
    zone: str
    method: str
    paired_count: int
    regression_mae: float | None = None
    bands_mae: float | None = None
    regression_missed: int | None = None
    bands_missed: int | None = None
    minimum: int = MIN_PAIRED_ORDERS
    reason: str = REASON_WINS

    @property
    def use_regression(self) -> bool:
        return self.method == REGRESSION

    @property
    def choice(self) -> str:
        return REGRESSION if self.use_regression else BASELINE

    @property
    def selected_label(self) -> str:
        return REGRESSION_LABEL if self.use_regression else BANDS_LABEL

    @property
    def selected_option(self) -> str:
        return REGRESSION_OPTION if self.use_regression else BANDS_OPTION

    @property
    def reason_text(self) -> str:
        return _REASON_TEXT.get(self.reason, self.reason)

    def evidence(self) -> dict[str, Any]:
        return {
            '作业区': self.zone,
            '分区': self.zone,
            '区': self.zone,
            'zone': self.zone,
            '选用估时': self.selected_option,
            '选择': self.selected_option,
            '选用套别': self.method,
            'method': self.method,
            'selected': self.method,
            'decision': self.reason,
            'use_model': self.use_regression,
            'choice': REGRESSION if self.use_regression else 'baseline',
            'selected_method': REGRESSION if self.use_regression else 'band',
            'estimator_kind': 'huber' if self.use_regression else 'duration_band',
            '选用模型': self.selected_label,
            '启用回归': self.use_regression,
            'paired_count': self.paired_count,
            'n': self.paired_count,
            '成对实绩': self.paired_count,
            '最低成对单量': self.minimum,
            'minimum': self.minimum,
            '回归MAE': _round(self.regression_mae),
            '分档MAE': _round(self.bands_mae),
            'regression_mae': self.regression_mae,
            'bands_mae': self.bands_mae,
            '回归平均绝对误差': _round(self.regression_mae),
            '分档平均绝对误差': _round(self.bands_mae),
            '回归漏查晚装车单': self.regression_missed,
            '分档漏查晚装车单': self.bands_missed,
            '模型漏查晚装车单': self.regression_missed,
            '基线漏查晚装车单': self.bands_missed,
            'regression_missed': self.regression_missed,
            'bands_missed': self.bands_missed,
            '回归漏查': self.regression_missed,
            '分档漏查': self.bands_missed,
            '模型漏查': self.regression_missed,
            '基线漏查': self.bands_missed,
            '决定码': self.reason,
            '原因码': self.reason,
            '决定原因': self.reason_text,
            '原因': self.reason_text,
        }

    # Compact alias expected by callers that treat the decision as a mapping.
    def as_dict(self) -> dict[str, Any]:
        return self.evidence()

    def __getitem__(self, key: str) -> Any:
        evidence = self.evidence()
        if key in evidence:
            return evidence[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def late_load_missed(estimate: float, actual: float, slack_min: float, *,
                     grace_minutes: float = 6.0, slack_known: bool = True,
                     diagnostic_boundary: bool = False) -> bool | None:
    """Return whether an estimate misses a ticket-specific late load.

    The default is the exact negation of the dispatch rule
    ``estimate > slack - grace``: ``estimate <= buffer < actual``.
    ``diagnostic_boundary=True`` uses the older replay-diagnostic inequality
    ``estimate < buffer <= actual``.
    """
    if not slack_known:
        return None
    buffer_minutes = slack_min - grace_minutes
    if diagnostic_boundary:
        return estimate < buffer_minutes <= actual
    return estimate <= buffer_minutes < actual


def decide_zone(zone: str, comparisons: tuple[ReplayComparison, ...] | list[ReplayComparison],
                *, minimum: int = MIN_PAIRED_ORDERS) -> ZoneChoice:
    # Sorting by ticket makes direct callers order-independent even before the
    # pipeline's canonical day/key sort runs.
    pairs = tuple(sorted(comparisons, key=lambda item: item.key))
    count = len(pairs)
    if count < minimum:
        return ZoneChoice(zone, BANDS, count, minimum=minimum,
                          reason=REASON_INSUFFICIENT if count else REASON_NO_REPLAY)

    regression_mae = fsum(item.regression_absolute_error for item in pairs) / count
    bands_mae = fsum(item.bands_absolute_error for item in pairs) / count
    regression_missed = sum(bool(item.regression_missed) for item in pairs)
    bands_missed = sum(bool(item.bands_missed) for item in pairs)

    if not regression_mae < bands_mae:
        reason = REASON_MAE
    elif regression_missed > bands_missed:
        reason = REASON_LATE_LOAD
    else:
        reason = REASON_WINS
    method = REGRESSION if reason == REASON_WINS else BANDS
    return ZoneChoice(zone, method, count, regression_mae, bands_mae,
                      regression_missed, bands_missed, minimum, reason)


def comparison_from_view(view: Any, bands_estimate: float) -> ReplayComparison:
    raw = view.raw
    slack_index = 5
    slack_known = 'slack_min' not in raw.missing
    return ReplayComparison(
        raw.key, raw.cohort, float(view.estimate), float(bands_estimate),
        float(raw.outcome), float(raw.input_values[slack_index]) - 6.0, slack_known,
    )


def comparisons_for_replay(checked_replay: Any, reference: Any) -> tuple[ReplayComparison, ...]:
    pairs = []
    for view in checked_replay:
        outcome = view.raw.outcome
        if outcome is None or not view.raw.observed:
            continue
        bands_estimate = pickup_estimate(reference, view.raw)
        pairs.append(comparison_from_view(view, bands_estimate))
    return tuple(sorted(pairs, key=lambda item: (item.zone, item.key)))


@dataclass(frozen=True, slots=True)
class ZonePolicy:
    choices: tuple[ZoneChoice, ...]
    reference: Any
    minimum: int = MIN_PAIRED_ORDERS

    @property
    def by_zone(self) -> dict[str, ZoneChoice]:
        return {choice.zone: choice for choice in self.choices}

    def get(self, zone: str, default=None) -> ZoneChoice | None:
        return self.by_zone.get(zone, default)

    def __getitem__(self, zone: str) -> ZoneChoice:
        return self.by_zone[zone]

    def __iter__(self):
        return iter(self.choices)

    def __len__(self) -> int:
        return len(self.choices)

    def items(self):
        return self.by_zone.items()

    def keys(self):
        return self.by_zone.keys()

    def values(self):
        return self.by_zone.values()

    def method_for(self, zone: str) -> str:
        choice = self.by_zone.get(zone)
        return REGRESSION if choice is not None and choice.use_regression else BANDS

    def use_regression(self, zone: str) -> bool:
        return self.method_for(zone) == REGRESSION

    def estimate(self, zone: str, wave: Any) -> float:
        if self.use_regression(zone):
            # Prediction/Assessment callers normally already have this value.
            return float(getattr(wave, 'estimate', getattr(wave, 'score')))
        return float(pickup_estimate(self.reference, wave))

    def select_assessment(self, assessment: Any) -> Any:
        zone = assessment.raw.cohort
        if self.use_regression(zone):
            return assessment
        selected = float(pickup_estimate(self.reference, assessment.raw))
        lower = assessment.lower
        upper = assessment.upper
        if lower is not None and upper is not None:
            shift = selected - assessment.estimate
            lower = max(0.0, float(lower) + shift)
            upper = max(0.0, float(upper) + shift)
        return replace(assessment, estimate=selected, lower=lower, upper=upper)

    def apply(self, assessments: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
        return tuple(self.select_assessment(item) for item in assessments)

    def evidence(self) -> list[dict[str, Any]]:
        return [choice.evidence() for choice in sorted(self.choices, key=lambda item: item.zone)]

    def mapping(self) -> dict[str, str]:
        return {choice.zone: choice.method for choice in sorted(self.choices, key=lambda item: item.zone)}

    def labels(self) -> dict[str, str]:
        return {choice.zone: choice.selected_option for choice in sorted(self.choices, key=lambda item: item.zone)}


def build_zone_policy(training: tuple[Any, ...] | list[Any], checked_replay: Any,
                      live_zones: tuple[str, ...] | list[str] = (), *,
                      minimum: int = MIN_PAIRED_ORDERS) -> ZonePolicy:
    reference = cohort_pick_times(training)
    comparisons = comparisons_for_replay(checked_replay, reference)
    grouped: dict[str, list[ReplayComparison]] = {}
    for pair in comparisons:
        grouped.setdefault(pair.zone, []).append(pair)

    zones = set(grouped).union(live_zones)
    choices = []
    for zone in sorted(zones):
        choice = decide_zone(zone, grouped.get(zone, ()), minimum=minimum)
        choices.append(choice)
    return ZonePolicy(tuple(choices), reference, minimum)


# Functional aliases for callers naming this as an estimator selector.
def select_zone_estimators(training, checked_replay, live_zones=(), **kwargs):
    return build_zone_policy(training, checked_replay, live_zones, **kwargs)


def choose_zone_estimators(training, checked_replay, live_zones=(), **kwargs):
    return build_zone_policy(training, checked_replay, live_zones, **kwargs)


def zone_estimator_choices(training, checked_replay, live_zones=(), **kwargs):
    return build_zone_policy(training, checked_replay, live_zones, **kwargs).evidence()


def build_policy_from_vectors(fit, training, replay, *, minimum: int = MIN_PAIRED_ORDERS) -> ZonePolicy:
    reference = cohort_pick_times(training)
    grouped: dict[str, list[ReplayComparison]] = {}
    for sample in replay:
        if sample.outcome is None or not sample.observed:
            continue
        regression, _ = score_kernel(fit.parameters, sample)
        bands = pickup_estimate(reference, sample)
        slack_known = 'slack_min' not in sample.missing
        pair = ReplayComparison(
            sample.key, sample.cohort, float(regression), float(bands), float(sample.outcome),
            float(sample.input_values[5]) - 6.0, slack_known,
        )
        grouped.setdefault(sample.cohort, []).append(pair)
    choices = tuple(decide_zone(zone, pairs, minimum=minimum) for zone, pairs in sorted(grouped.items()))
    return ZonePolicy(choices, reference, minimum)
