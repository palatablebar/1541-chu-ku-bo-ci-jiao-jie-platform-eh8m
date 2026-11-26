"""Alias for applying replay-selected zone estimates to dispatch."""

from .aisle_dispatch.service import apply_selected_estimates, apply_zone_policy
from .zone_selection import (
    MIN_PAIRED_ORDERS,
    ZonePolicy,
    build_zone_policy,
    decide_zone,
)


def select_and_apply(assessments, training, checked_replay, live_zones=(), *,
                     minimum=MIN_PAIRED_ORDERS):
    policy = build_zone_policy(training, checked_replay, live_zones, minimum=minimum)
    return policy.apply(assessments), policy


apply_estimator_policy = apply_zone_policy
route_with_selected_estimators = apply_zone_policy

__all__ = (
    'apply_zone_policy',
    'apply_selected_estimates',
    'apply_estimator_policy',
    'route_with_selected_estimators',
    'select_and_apply',
    'build_zone_policy',
    'decide_zone',
    'ZonePolicy',
)
