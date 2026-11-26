"""Public alias for per-zone replay estimator selection.

The implementation lives with duration models; this module gives callers a
stable top-level name for the cross-stage policy.
"""

from .handoff_duration.zone_choice import (
    BANDS,
    MIN_PAIRED_ORDERS,
    REGRESSION,
    ZoneChoice,
    ZonePolicy,
    build_policy_from_vectors,
    build_zone_policy,
    decide_zone,
    late_load_missed,
    select_zone_estimators,
    choose_zone_estimators,
    zone_estimator_choices,
)

MIN_PAIRED_SAMPLES = MIN_PAIRED_ORDERS
select_estimators = build_zone_policy
choose_estimators = build_zone_policy
evaluate_zones = build_zone_policy
compare_zones = zone_estimator_choices

__all__ = (
    'BANDS', 'REGRESSION', 'MIN_PAIRED_ORDERS', 'MIN_PAIRED_SAMPLES',
    'ZoneChoice', 'ZonePolicy', 'build_zone_policy', 'build_policy_from_vectors',
    'decide_zone', 'late_load_missed', 'select_zone_estimators',
    'choose_zone_estimators', 'zone_estimator_choices', 'select_estimators',
    'choose_estimators', 'evaluate_zones', 'compare_zones',
)
