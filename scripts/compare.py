"""Compare two runoff CSVs written by run_forcing / run_python_reference.

    python scripts/compare.py <reference.csv> <candidate.csv>
"""

import csv
import math
import sys


def load(path):
    with open(path, newline='') as f:
        return [float(r['runoff_m_per_h']) for r in csv.DictReader(f)]


def main(ref_path, cand_path):
    ref, cand = load(ref_path), load(cand_path)
    if len(ref) != len(cand):
        raise SystemExit(f'length mismatch: {len(ref)} vs {len(cand)}')
    first = next((i for i, v in enumerate(ref) if v != 0.0), None)
    first_c = next((i for i, v in enumerate(cand) if v != 0.0), None)
    print(f'steps: {len(ref)}, first non-zero step: reference {first}, candidate {first_c}')
    pairs = [(r, c) for r, c in zip(ref, cand)]
    abs_err = [abs(r - c) for r, c in pairs]
    rel_err = [abs(r - c) / abs(r) for r, c in pairs if abs(r) > 1e-9]
    worst = max(range(len(abs_err)), key=abs_err.__getitem__)
    mean_ref = sum(ref) / len(ref)
    nse = 1 - sum((r - c) ** 2 for r, c in pairs) / max(sum((r - mean_ref) ** 2 for r in ref), 1e-300)
    print(f'max abs error: {abs_err[worst]:.3e} (step {worst}: ref {ref[worst]:.6e}, cand {cand[worst]:.6e})')
    if rel_err:
        rel_sorted = sorted(rel_err)
        print(
            f'relative error: median {rel_sorted[len(rel_sorted) // 2]:.3e}, '
            f'p99 {rel_sorted[int(len(rel_sorted) * 0.99)]:.3e}, max {rel_sorted[-1]:.3e}',
        )
    print(f'total runoff: reference {sum(ref):.6e}, candidate {sum(cand):.6e}')
    print(f'NSE of candidate vs reference: {nse:.9f}')
    if any(math.isnan(c) for c in cand):
        print('candidate contains NaN')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
