"""Naming alias for zone estimator selection."""

from .zone_selection import *  # noqa: F401,F403
from .zone_selection import build_zone_policy, decide_zone

select_zone_estimator = build_zone_policy
zone_estimator_policy = build_zone_policy
choose_zone_estimator = decide_zone

__all__ = ('build_zone_policy', 'decide_zone', 'select_zone_estimator',
           'zone_estimator_policy', 'choose_zone_estimator')
