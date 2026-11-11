"""Synthetic mixed-loading attributes for RS/CSD 3D-packing benchmarks.

Research simulator only: rules and synthetic properties are illustrative, NOT
real dangerous-goods compatibility or safety guidance.

Colab:
  from src.data.augment import load_data, augment_dataset, validate_dataset
  raw = load_data('/content/rs_labeled.pkl')
  enriched = augment_dataset(raw, dataset_name='rs', seed=42)
  print(validate_dataset(raw, enriched))

CLI:
  python -m src.data.augment --input rs_labeled.pkl --output rs_v3.pkl --name rs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORY_MAP = {'위험물': 0, '식품': 1, '화학품': 2, '일반': 3}
RISK_NAMES = ('compatible', 'low', 'medium', 'high', 'forbidden')
RELATION_TYPES = ('odor', 'contamination', 'physical_damage', 'temperature', 'general')
# Category-level bans inherited from the supplied data_utils.py; hypothetical.
DIRECT_FORBIDDEN = {frozenset(x) for x in (
    ('위험물', '식품'), ('위험물', '화학품'), ('화학품', '식품'))}


def load_data(path: str | Path) -> list:
    """Only unpickle trusted files (pickle is not safe for untrusted input)."""
    with open(path, 'rb') as f:
        result = pickle.load(f)
    if not isinstance(result, list) or not all(isinstance(s, list) for s in result):
        raise ValueError('Expected list of item-sequence lists')
    return result


def save_data(data: list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _rng(seed: int, dataset_name: str, seq_idx: int, item_idx: int) -> random.Random:
    # No dependence on global random state or augmentation call order.
    key = f'{seed}|{dataset_name}|{seq_idx}|{item_idx}'.encode('utf-8')
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))


def augment_item(item: dict, *, dataset_name: str, seq_idx: int,
                 item_idx: int, seed: int = 42) -> dict:
    """Preserve original fields, add deterministic scenario-specific attributes.

    item_uid identifies an occurrence, not a real SKU: the source has no SKU ID.
    Dimension units are unknown; density/weight are *simulation units*.
    """
    out = dict(item)
    cat = str(out.get('category'))
    if cat not in CATEGORY_MAP:
        raise ValueError(f'Unknown category: {cat!r}')
    dims = out.get('size')
    if not isinstance(dims, (list, tuple)) or len(dims) != 3 or not all(
            isinstance(x, (int, float)) and math.isfinite(float(x)) and x > 0 for x in dims):
        raise ValueError(f'Invalid size at {seq_idx}:{item_idx}: {dims!r}')
    if 'category_id' in out and int(out['category_id']) != CATEGORY_MAP[cat]:
        raise ValueError(f'Category ID mismatch at {seq_idx}:{item_idx}')
    r = _rng(seed, dataset_name, seq_idx, item_idx)
    out['category'] = cat
    out['item_uid'] = f'{dataset_name}:{seq_idx}:{item_idx}'
    out['arrival_index'] = item_idx
    out['volume'] = float(math.prod(dims))
    # Weight units are arbitrary until the benchmark's physical units are known.
    density = {'위험물': (0.25, 1.1), '식품': (0.15, 0.85),
               '화학품': (0.3, 1.3), '일반': (0.1, 0.95)}[cat]
    out['weight_sim'] = round(out['volume'] * r.uniform(*density), 4)
    out['sealed'] = r.random() < {'위험물': .75, '식품': .65,
                                 '화학품': .78, '일반': .7}[cat]
    out['leak_risk'] = round(r.uniform(0.3, 0.95) if cat in ('화학품', '위험물')
                             else r.uniform(0, 0.2), 3)
    out['fragile'] = r.random() < {'위험물': .15, '식품': .3,
                                  '화학품': .2, '일반': .35}[cat]
    out['max_supported_weight_sim'] = round(out['weight_sim'] *
        (r.uniform(0.0, 0.6) if out['fragile'] else r.uniform(1.5, 5.0)), 4)
    out['temperature_class'] = r.choices(
        ['ambient', 'chilled', 'frozen'],
        weights=([.1, .7, .2] if cat == '식품' else [.92, .07, .01]))[0]
    out['odor_profile'] = (r.choices(['release', 'absorb', 'neutral'],
                              weights=[.25, .25, .5])[0] if cat == '식품'
                              else r.choices(['release', 'neutral'],
                              weights=[.2, .8])[0])
    out['is_electronic'] = cat == '일반' and r.random() < .3
    out['hazard_type'] = (r.choice(['flammable', 'toxic', 'other']) if cat == '위험물'
        else r.choice(['oxidizing', 'corrosive', 'other']) if cat == '화학품'
        else 'none')
    # Preserve original hazard_level: it is category-level, not pair risk.
    return out


def augment_dataset(data: list, *, dataset_name: str, seed: int = 42) -> list:
    return [[augment_item(item, dataset_name=dataset_name, seq_idx=i,
                          item_idx=j, seed=seed) for j, item in enumerate(seq)]
            for i, seq in enumerate(data)]


def _signal_map(a: dict, b: dict) -> dict[str, float]:
    odor = float((a['odor_profile'] == 'release' and b['odor_profile'] == 'absorb') or
                 (b['odor_profile'] == 'release' and a['odor_profile'] == 'absorb'))
    if odor and a['sealed'] and b['sealed']:
        odor *= .25
    contamination = max(float(a['leak_risk']), float(b['leak_risk'])) * (
        .3 if a['sealed'] and b['sealed'] else 1.0)
    physical = float(a['fragile'] or b['fragile']) * .55
    temp = float(a['temperature_class'] != b['temperature_class']) * .7
    return {'odor': round(odor, 3), 'contamination': round(contamination, 3),
            'physical_damage': round(physical, 3), 'temperature': round(temp, 3),
            'general': 0.0}


def check_relation_v3(a: dict, b: dict, *, separated: bool = False) -> dict[str, Any]:
    """Symmetric pair relation; does not replace geometry/weight checks.

    Returns base risk (intrinsic relationship), contextual risk (packaging and
    separation), relation channels and hard/conditional decision.
    Separation is a *simulated* protective condition, not a measured distance.
    """
    ca, cb = str(a['category']), str(b['category'])
    if frozenset((ca, cb)) in DIRECT_FORBIDDEN:
        return {'base_risk': 4, 'base_label': 'forbidden',
                'context_risk': 4, 'context_label': 'forbidden',
                'channels': ['general'], 'signals': {'general': 1.0},
                'allowed': False, 'reason': 'category_direct_forbidden'}
    signals = _signal_map(a, b)
    # Rule examples: electronics near certain chemicals/hazards; odor transfer.
    electronics = bool(a['is_electronic'] or b['is_electronic'])
    sensitive = ((ca == '위험물' or cb == '위험물') or
                 (ca == '화학품' and a['hazard_type'] in ('oxidizing', 'corrosive')) or
                 (cb == '화학품' and b['hazard_type'] in ('oxidizing', 'corrosive')))
    if electronics and sensitive:
        signals['contamination'] = max(signals['contamination'], .75)
    channels = [k for k, v in signals.items() if v >= .3 and k != 'general'] or ['general']
    # Intrinsic risk is evaluated without packaging mitigation, for clear target separation.
    intrinsic_odor = float((a['odor_profile'] == 'release' and b['odor_profile'] == 'absorb') or
                           (b['odor_profile'] == 'release' and a['odor_profile'] == 'absorb'))
    intrinsic_leak = max(float(a['leak_risk']), float(b['leak_risk']))
    score = max(intrinsic_odor * .85, intrinsic_leak * .7,
                .65 if electronics and sensitive else 0,
                .45 if a['fragile'] or b['fragile'] else 0,
                .6 if a['temperature_class'] != b['temperature_class'] else 0)
    base = 3 if score >= .8 else 2 if score >= .55 else 1 if score >= .3 else 0
    context_score = max(signals.values())
    if separated and base < 4:
        # Separation mitigates odor/contact/physical, not temperature differences.
        context_score = max(signals['odor'] * .35,
                            signals['contamination'] * .45,
                            signals['physical_damage'] * .5,
                            signals['temperature'])
    contextual = 3 if context_score >= .8 else 2 if context_score >= .55 else 1 if context_score >= .3 else 0
    # High risk is always blocked; medium is allowed only with separation or both sealed,
    # and no temperature mismatch. This is a chosen experimental rule.
    medium_ok = (separated or (a['sealed'] and b['sealed'])) and (
        a['temperature_class'] == b['temperature_class'])
    allowed = contextual < 2 or (contextual == 2 and medium_ok)
    return {'base_risk': base, 'base_label': RISK_NAMES[base],
            'context_risk': contextual, 'context_label': RISK_NAMES[contextual],
            'channels': channels, 'signals': signals,
            'allowed': bool(allowed),
            'reason': 'allowed' if allowed else ('high_context_risk' if contextual >= 3
                      else 'medium_requires_protection')}


def relation_edges(seq: list, *, include_compatible: bool = True):
    """Yield undirected pair labels lazily; avoid materializing all O(n²) edges."""
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            relation = check_relation_v3(seq[i], seq[j])
            if include_compatible or relation['base_risk'] > 0:
                yield i, j, relation


def validate_dataset(original: list, augmented: list, *, sample_pairs: int = 5000,
                     seed: int = 42) -> dict:
    if len(original) != len(augmented):
        raise AssertionError('Sequence count changed')
    cats, temps, risk, channel_counts = Counter(), Counter(), Counter(), Counter()
    n = 0
    for i, (before, after) in enumerate(zip(original, augmented)):
        if len(before) != len(after):
            raise AssertionError(f'Item count changed at sequence {i}')
        for j, (a, b) in enumerate(zip(before, after)):
            for field in ('size', 'category', 'category_id', 'hazard_level'):
                if field in a and (list(a[field]) if field == 'size' else str(a[field])) != (
                        list(b[field]) if field == 'size' else str(b[field])):
                    raise AssertionError(f'Original field changed: {i}:{j}:{field}')
            if b['arrival_index'] != j or b['weight_sim'] <= 0:
                raise AssertionError(f'Invalid derived fields at {i}:{j}')
            cats[b['category']] += 1
            temps[b['temperature_class']] += 1
            n += 1
    rng = random.Random(seed)
    eligible = [i for i, s in enumerate(augmented) if len(s) >= 2]
    for _ in range(sample_pairs):
        if not eligible:
            break
        seq = augmented[rng.choice(eligible)]
        a, b = rng.sample(seq, 2)
        ab, ba = check_relation_v3(a, b), check_relation_v3(b, a)
        if (ab['base_risk'], ab['context_risk'], ab['allowed']) != (
                ba['base_risk'], ba['context_risk'], ba['allowed']):
            raise AssertionError('Asymmetric pair rule')
        risk[ab['base_label']] += 1
        channel_counts.update(ab['channels'])
    return {'sequences': len(augmented), 'items': n,
            'category_counts': dict(cats), 'temperature_counts': dict(temps),
            'sampled_pairs': sum(risk.values()), 'pair_base_risk_counts': dict(risk),
            'pair_channel_counts': dict(channel_counts),
            'preserved_original_fields': True,
            'note': 'Synthetic rules; not real safety labels. Pair statistics sampled.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--name', required=True, help='e.g. rs or csd')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample-pairs', type=int, default=5000)
    args = p.parse_args()
    original = load_data(args.input)
    augmented = augment_dataset(original, dataset_name=args.name, seed=args.seed)
    report = validate_dataset(original, augmented, sample_pairs=args.sample_pairs)
    save_data(augmented, args.output)
    report_path = Path(args.output).with_suffix('.validation.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Saved: {args.output}\nValidation: {report_path}')


if __name__ == '__main__':
    main()
