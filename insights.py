"""Analytics-only helpers. These never alter model features or training."""
import pandas as pd

REQUIRED = ['user_id', 'problem_id', 'skill_name', 'correct']

def validate_records(records):
    missing = set(REQUIRED) - set(records.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    data = records[REQUIRED].copy()
    data['correct'] = pd.to_numeric(data['correct'], errors='coerce')
    for col in REQUIRED[:-1]:
        data[col] = data[col].astype('string').str.strip()
    valid = data['correct'].isin([0, 1])
    for col in REQUIRED[:-1]:
        valid &= data[col].notna() & data[col].ne('')
    clean = data.loc[valid].copy()
    clean['correct'] = clean['correct'].astype(int)
    return clean, int((~valid).sum())

def summarize(data):
    return data.groupby('skill_name').agg(attempts=('correct', 'size'), correct_rate=('correct', 'mean'))

def action(count, rate, minimum, threshold):
    if count < minimum:
        return 'Collect more evidence'
    if rate < threshold:
        return 'Review misconceptions; offer guided practice'
    return 'Continue practice; monitor progress'

def comparison(before, after, minimum, threshold):
    table = summarize(before).join(summarize(after), how='outer', lsuffix='_before', rsuffix='_after')
    for suffix in ['before', 'after']:
        table['attempts_' + suffix] = table['attempts_' + suffix].fillna(0).astype(int)
        table['action_' + suffix] = [action(n, r, minimum, threshold) for n, r in zip(table['attempts_' + suffix], table['correct_rate_' + suffix])]
    table['added_attempts'] = table.attempts_after - table.attempts_before
    table['change_pp'] = (table.correct_rate_after - table.correct_rate_before) * 100
    table['recommendation_changed'] = table.action_before != table.action_after
    return table.reset_index()
