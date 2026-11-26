from collections import defaultdict
from math import fabs, fsum
from statistics import mean, median

from .huber_model import score_kernel


def picker_band(wave):
    lines, _, pickers, *_ = wave.input_values
    return int(lines/pickers >= 14), wave.cohort


def cohort_pick_times(waves):
    bands = defaultdict(list)
    for previous in waves:
        bands[picker_band(previous)].append(previous.outcome)
    fitted = {segment:median(times) for segment,times in bands.items() if len(times)>=4}
    return fitted, median(previous.outcome for previous in waves)


def pickup_estimate(reference, wave):
    fitted, overall = reference
    return fitted.get(picker_band(wave), overall)


def baseline_review(fit, training, replay):
    convention = cohort_pick_times(training)
    production = defaultdict(list)
    for sample in sorted(replay, key=lambda item: (item.cohort, item.key)):
        if not sample.observed or sample.outcome is None:
            continue
        estimate, _ = score_kernel(fit.parameters,sample)
        baseline = pickup_estimate(convention, sample)
        actual = sample.outcome
        loss, prior = abs(actual-estimate),abs(actual-baseline)
        slack_known = 'slack_min' not in sample.missing
        buffer_minutes = sample.input_values[5] - 6
        production[sample.cohort].append(
            (loss, prior, estimate-actual, baseline-actual, estimate, baseline,
             actual, buffer_minutes, slack_known)
        )
    reports = []
    for sector, comparison in sorted(production.items()):
        (model_loss, base_loss, error, original, model_estimates, baseline_estimates,
         actuals, buffers, slack_known) = zip(*comparison, strict=True)
        paired = tuple(old-new for old,new in zip(base_loss,model_loss,strict=True))
        n = len(comparison)
        model_mae = mean(model_loss)
        baseline_mae = mean(base_loss)
        model_missed = sum(
            known and estimate <= buffer_minutes < actual
            for estimate, actual, buffer_minutes, known in zip(
                model_estimates, actuals, buffers, slack_known, strict=True)
        )
        baseline_missed = sum(
            known and estimate <= buffer_minutes < actual
            for estimate, actual, buffer_minutes, known in zip(
                baseline_estimates, actuals, buffers, slack_known, strict=True)
        )
        if n < 6:
            choice, reason, use_regression = '分档估时', 'INSUFFICIENT_PAIRED_ORDERS', False
        elif not model_mae < baseline_mae:
            choice, reason, use_regression = '分档估时', 'MAE_NOT_LOWER', False
        elif model_missed > baseline_missed:
            choice, reason, use_regression = '分档估时', 'LATE_LOAD_MISS_OVERRIDE', False
        else:
            choice, reason, use_regression = '回归估时', 'REGRESSION_REPLAY_WINS', True
        reports.append({'回放组':sector,'作业区':sector,'分区':sector,'zone':sector,'实绩数':n,'成对实绩':n,'基线':'人均行数分档中位','估计代价':round(model_mae,4),
                        '业务基线代价':round(baseline_mae,4),'模型偏误中位':round(median(error),4),
                        '参照偏误中位':round(median(original),4),'同单误差减少中位':round(median(paired),4),
                        '改善单数':sum(delta>0 for delta in paired),'恶化单数':sum(delta<0 for delta in paired),
                        '选用估时':choice, '选择':choice, 'decision':reason,
                        '选用套别':'regression' if use_regression else 'bands',
                        'selected':'regression' if use_regression else 'bands',
                        '启用回归':use_regression, 'use_regression':use_regression,
                        'use_model':use_regression, '决定码':reason,
                        '回归MAE':round(model_mae,4), '分档MAE':round(baseline_mae,4),
                        '模型MAE':round(model_mae,4), '基线MAE':round(baseline_mae,4),
                        '回归漏查晚装车单':model_missed, '分档漏查晚装车单':baseline_missed,
                        '模型漏查晚装车单':model_missed, '基线漏查晚装车单':baseline_missed,
                        '模型漏查':model_missed, '基线漏查':baseline_missed})
    return reports


def select_zone_models(fit, training, replay, *, minimum=6):
    """Return structured zone choices from feature vectors and a fitted model."""
    from .zone_choice import build_policy_from_vectors
    return build_policy_from_vectors(fit, training, replay, minimum=minimum)


def choose_zone_estimators(fit, training, replay, *, minimum=6):
    return select_zone_models(fit, training, replay, minimum=minimum)


def zone_estimator_report(fit, training, replay, *, minimum=6):
    return select_zone_models(fit, training, replay, minimum=minimum).evidence()
