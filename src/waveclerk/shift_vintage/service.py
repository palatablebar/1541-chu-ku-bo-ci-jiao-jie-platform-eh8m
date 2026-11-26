from collections import Counter, defaultdict
from dataclasses import replace

from ..pick_manifests.contracts import WaveclerkRecord
from .readiness import formation_tail
from .contracts import Chronology, FoldSummary, Span, WindowRule


def assign_fold(record, rule):
    if record.day in rule.live and record.day <= rule.as_of and record.received <= rule.as_of:
        return 'live', 'OPEN_RUN'
    for name, span in rule.folds():
        if record.day not in span:
            continue
        if record.received >= span.end or record.result_day is None or record.result_day >= span.end:
            return None, 'RESULT_NOT_MATURE'
        if record.result_day < record.day:
            return None, 'REVERSED_RESULT_TIME'
        if record.outcome is None:
            return None, 'MISSING_OUTCOME'
        if not record.observed and not rule.allow_censor:
            return None, 'CENSORED_OUTCOME'
        return name, 'LOCKED_HISTORY'
    return None, 'EMBARGO_OR_UNSCHEDULED'


def route_history(records, rule):
    panels = {name: [] for name, _ in rule.folds()}
    panels['live'] = []
    lost = []
    for record in records:
        fold, reason = assign_fold(record, rule)
        if fold is None:
            lost.append((record.key, reason))
        else:
            panels[fold].append(record)
    folds = {}
    for name, values in panels.items():
        folds[name] = tuple(sorted(values, key=lambda item: (item.day, item.key)))
    return Chronology(**folds, excluded=tuple(sorted(lost)))


def build_windows(records, rule=None):
    active = rule or WindowRule.standard(allow_censor=False)
    views = route_history(records, active)
    views.check_conservation(len(records))
    return views


def _span_for_day(day, rule):
    for name, span in rule.folds():
        if day in span:
            return name, span
    return None, None


def _record_vintage(records, cutoff):
    """Select revisions from already-decoded WaveclerkRecord objects."""
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        if record.received <= cutoff:
            grouped[record.key].append((index, record))
    selected = []
    for versions in grouped.values():
        highest = max(record.revision for _, record in versions)
        current = [(index, record) for index, record in versions if record.revision == highest]
        if len({record.signature() for _, record in current}) > 1:
            continue
        _, record = min(current, key=lambda item: (item[1].received, item[0]))
        selected.append(record)
    return tuple(sorted(selected, key=lambda record: (record.day, record.key)))


def _vintage_entries(documents, cutoff):
    first = next(iter(documents), None)
    if isinstance(first, WaveclerkRecord):
        return {row.key: row for row in _record_vintage(documents, cutoff)}
    from ..pick_manifests.service import load_samples
    return {row.key: row for row in load_samples(documents, cutoff).entries}


def build_vintage_windows(documents, current, rule=None):
    """Select the newest revision available at each historical fold cutoff.

    ``current`` is the intake result at the shift cutoff.  Its key set is the
    conservation boundary for the audit ledger.  Raw documents are decoded again
    at each half-open fold endpoint, so a correction arriving after that
    endpoint cannot replace the revision that was available then.
    """
    active = rule or WindowRule.standard(allow_censor=False)
    if hasattr(current, 'entries'):
        current = current.entries
    current = tuple(current)
    source_documents = tuple(documents)
    current_windows = route_history(current, active)
    available = {}
    for name, span in active.folds():
        available[name] = _vintage_entries(source_documents, span.end - 1)

    panels = {name: [] for name, _ in active.folds()}
    panels['live'] = list(current_windows.live)
    excluded = []

    for record in current:
        assigned, _ = assign_fold(record, active)
        if assigned == 'live':
            continue

        fold_name, _span = _span_for_day(record.day, active)
        candidate = available.get(fold_name, {}).get(record.key) if fold_name else None
        if candidate is not None:
            candidate_fold, reason = assign_fold(candidate, active)
            if candidate_fold in panels and candidate_fold != 'live':
                panels[candidate_fold].append(candidate)
                continue
            excluded.append((record.key, reason))
            continue

        # Preserve the established reason for records that never had a mature,
        # available vintage in their scheduled fold.
        reason = dict(current_windows.excluded).get(record.key)
        if reason is None:
            reason = 'EMBARGO_OR_UNSCHEDULED' if fold_name is None else 'VERSION_NOT_AVAILABLE'
        excluded.append((record.key, reason))

    folds = {}
    for name, values in panels.items():
        folds[name] = tuple(sorted(values, key=lambda item: (item.day, item.key, item.revision)))
    views = Chronology(**folds, excluded=tuple(sorted(excluded)))
    views.check_conservation(len(current))
    return views


# Descriptive aliases used by callers discussing point-in-time revisions.
build_point_in_time_windows = build_vintage_windows
build_revision_windows = build_vintage_windows


def partition_summary(windows):
    rows = []
    for name, records in windows.partitions():
        cohorts = Counter(record.cohort for record in records)
        window = FoldSummary(name, len(records), len(cohorts),
                             sum(not row.observed for row in records),
                             records[0].day if records else None, records[-1].day if records else None)
        rows.append(window)
    return tuple(rows)


def window_evidence(windows):
    names = {'train': '拟合', 'calibration': '校准', 'replay': '回放', 'live': '当班'}
    evidence = [
        {'时段': names[part.name], '单据': part.count, '分组': part.cohort_count,
         '截尾': part.censored, '首日': part.first_day, '末日': part.last_day}
        for part in partition_summary(windows)
    ]
    for code, count in sorted(Counter(reason for _, reason in windows.excluded).items()):
        evidence.append({'时段': code, '单据': count, '分组': None, '截尾': None, '首日': None, '末日': None})
    return evidence


def model_inputs(windows):
    if len(windows.train) < 16:
        raise ValueError('至少需要 16 条在拟合期内成熟的记录')
    if not windows.calibration:
        raise ValueError('校准期没有可用结果')
    if not windows.replay:
        raise ValueError('回放期没有可用结果')
    if not windows.live:
        raise ValueError('当班时段没有可用单据')
    return tuple(row.key for row in windows.train)


def readiness_evidence(windows, as_of=96):
    valid = formation_tail(windows, replace(WindowRule.standard(), as_of=as_of))
    stages = ('train','calibration','replay','live')
    return sorted(valid,key=lambda row:(stages.index(row['交班段']), -row['有效单'], row['来源']))


def vintage_revision_evidence(windows, current):
    """List tickets where a historical fold uses an older available revision."""
    names = {'train': '拟合', 'calibration': '校准', 'replay': '回放'}
    current_by_key = {row.key: row for row in current}
    rows = []
    for fold, label in names.items():
        for selected in sorted(getattr(windows, fold), key=lambda row: row.key):
            latest = current_by_key.get(selected.key)
            if latest is None or selected.revision == latest.revision:
                continue
            rows.append({
                '时段': label,
                '单号': selected.key,
                '作业区': selected.cohort,
                '窗口采用版次': selected.revision,
                '窗口采用到站日': selected.received,
                '当班最高版次': latest.revision,
                '当班版次到站日': latest.received,
            })
    return rows
