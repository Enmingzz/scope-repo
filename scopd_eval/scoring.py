"""Aggregate correctness without erasing dataset-specific paired scoring."""
from collections import defaultdict
from statistics import mean


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset']].append(row)
    reports = {}
    for dataset, items in groups.items():
        report = {'rows': len(items), 'accuracy_percent': 100 * mean(r['correct'] for r in items),
                  'deterministic_match_percent': 100 * mean(r['exact_match'] for r in items)}
        categories = defaultdict(list)
        for row in items:
            categories[str(row.get('metadata', {}).get('category', 'all'))].append(row['correct'])
        report['category_accuracy_percent'] = {k: 100 * mean(v) for k, v in categories.items()}
        if dataset == 'MME':
            pairs = defaultdict(list)
            for row in items:
                meta = row['metadata']
                pairs[(meta['category'], meta['image_path'])].append(row['correct'])
            if any(len(v) != 2 for v in pairs.values()):
                report['mme_score'] = None
                report['note'] = 'Incomplete MME image pairs: no official-style total reported'
            else:
                by_category = defaultdict(list)
                for (category, _), values in pairs.items():
                    by_category[category].append(all(values))
                report['mme_category_score'] = {k: 100 * (mean(categories[k]) + mean(v))
                                                for k, v in by_category.items()}
                report['mme_score'] = sum(report['mme_category_score'].values())
        if dataset == 'MMVP':
            pairs = defaultdict(list)
            for row in items:
                idx = int(row['sample_id'])
                if idx < 1:
                    raise ValueError('MMVP official indices start at 1')
                pairs[(idx - 1) // 2].append(row['correct'])
            report['pair_accuracy_percent'] = (100 * mean(all(v) for v in pairs.values())
                                               if all(len(v) == 2 for v in pairs.values()) else None)
        if dataset == 'HallusionBench':
            figures, questions = defaultdict(list), defaultdict(list)
            for row in items:
                parts = str(row['sample_id']).split('_')
                if len(parts) < 6:
                    raise ValueError('HallusionBench IDs lack set/figure/question fields')
                cat = row['metadata']['l2-category']
                figures[(cat, parts[3], parts[4])].append(row['correct'])
                questions[(cat, parts[3], parts[5])].append(row['correct'])
            report.update(aAcc=report['accuracy_percent'],
                          fAcc=100 * mean(all(v) for v in figures.values()),
                          qAcc=100 * mean(all(v) for v in questions.values()))
        if dataset == 'POPE':
            report['metric'] = 'accuracy; not F1'
        if dataset == 'CVBench':
            two_d, three_d = defaultdict(list), []
            for row in items:
                meta = row['metadata']
                if meta['split'] == '2D':
                    two_d[meta['source']].append(row['correct'])
                elif meta['split'] == '3D':
                    three_d.append(row['correct'])
                else:
                    raise ValueError('CVBench split must be 2D or 3D')
            report['2d_source_macro_percent'] = 100 * mean(mean(v) for v in two_d.values()) if two_d else None
            report['3d_accuracy_percent'] = 100 * mean(three_d) if three_d else None
            report['cvbench_macro_percent'] = (mean([report['2d_source_macro_percent'], report['3d_accuracy_percent']])
                                                if two_d and three_d else None)
        reports[dataset] = report
    return reports
