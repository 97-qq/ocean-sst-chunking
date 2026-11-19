"""Compare fixed and adaptive Zarr replica policies on chronological synthetic workloads."""

import csv
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import zarr

from replay import (
    BASE_CHUNKS, HOT, MIB, NEW_HOT, ROOT, SERIES_CHUNKS, SHAPE, TILE_SIDE, TILES,
    CountingStore, build_replica, check_query, chunk_cost, create_array,
    spatial_parent, synthetic_sst, train_ranking,
)


WINDOWS = 36
BUDGET = 8
CHANGE = 12
SEEDS = range(8201, 8231)
SCENARIOS = ('Stable', 'Persistent shift', 'One-window bursts', 'Alternating', 'Uniform')
POLICIES = ('Original', 'Fixed', 'Rolling', 'Confirm only', 'Payback only', 'Guarded', 'Full')
COLORS = {'Fixed': '#59677a', 'Rolling': '#c05a42', 'Guarded': '#137b75', 'Full': '#ad873e'}
BASE_POINT = chunk_cost(BASE_CHUNKS, point=True)[1]
REPLICA_POINT = chunk_cost(SERIES_CHUNKS, point=True)[1]
MAP_BYTES = chunk_cost(BASE_CHUNKS, point=False)[1]
SAVING_PER_HIT = BASE_POINT - REPLICA_POINT
ALL_TILES = frozenset(range(TILES))


def make_trace(seed, scenario, points=32, burst_duration=None):
    """Generate separate calibration and evaluation streams for one experiment."""
    calibration, evaluation = np.random.SeedSequence(seed).spawn(2)
    initial = frozenset(train_ranking(np.random.default_rng(calibration))[:BUDGET])
    rng = np.random.default_rng(evaluation)
    random_tiles = rng.integers(TILES, size=(WINDOWS, points))
    hotspot_draw = rng.integers(BUDGET, size=(WINDOWS, points))
    concentrated = rng.random((WINDOWS, points)) < .85
    windows = []
    for t in range(WINDOWS):
        if burst_duration is not None:
            shifted = CHANGE <= t < CHANGE + burst_duration
        else:
            shifted = (scenario == 'Persistent shift' and t >= CHANGE
                       or scenario == 'One-window bursts' and t in (6, 12, 18, 24, 30)
                       or scenario == 'Alternating' and t % 2 == 1)
        hotspot = NEW_HOT if shifted else HOT
        chosen = random_tiles[t] if scenario == 'Uniform' else np.where(
            concentrated[t], hotspot[hotspot_draw[t]], random_tiles[t]
        )
        windows.append(np.bincount(chosen, minlength=TILES))
    return initial, np.array(windows)


def top_tiles(counts):
    return frozenset(np.lexsort((np.arange(TILES), -counts))[:BUDGET])


def hits(counts, tiles):
    return int(counts[list(tiles)].sum()) if tiles else 0


def update_cost(additions):
    parents = {spatial_parent(tile) for tile in additions}
    return len(parents) * BASE_POINT, len(additions) * REPLICA_POINT


def choose_replica(policy, current, previous):
    if not len(previous) or policy in ('Original', 'Fixed', 'Full'):
        return current
    candidate = top_tiles(previous[-1])
    if policy == 'Rolling':
        return candidate
    confirmation_windows = 1 if policy == 'Payback only' else 2
    if len(previous) < confirmation_windows or candidate == current:
        return current
    conservative_gain = min(hits(window, candidate) - hits(window, current)
                            for window in previous[-confirmation_windows:])
    cost = 0 if policy == 'Confirm only' else sum(update_cost(candidate - current))
    if conservative_gain * SAVING_PER_HIT > cost:
        return candidate
    return current


def replay(seed, scenario, initial, workload, policy):
    planned = {'Original': frozenset(), 'Full': ALL_TILES}.get(policy, initial)
    resident = frozenset()
    cumulative = 0
    rows = []
    for t, counts in enumerate(workload):
        planned = choose_replica(policy, planned, workload[:t])
        additions = planned - resident
        # The budget applies to the active layout after each replacement.
        assert len(planned) == {'Original': 0, 'Full': TILES}.get(policy, BUDGET)
        build_read, build_write = update_cost(additions)
        point_queries = int(counts.sum())
        baseline = point_queries * BASE_POINT + (point_queries // 4) * MAP_BYTES
        n_hits = hits(counts, planned)
        query = baseline - n_hits * SAVING_PER_HIT
        net = baseline - query - build_read - build_write
        cumulative += net
        rows.append(dict(seed=seed, scenario=scenario, policy=policy, window=t + 1,
                         point_queries=point_queries, point_hits=n_hits,
                         baseline_bytes=baseline, query_bytes=query,
                         build_read_bytes=build_read, build_write_bytes=build_write,
                         tiles_added=len(additions), net_saved_bytes=net,
                         cumulative_net_saved_bytes=cumulative))
        resident = planned
    return rows


def incremental_audit():
    cases = {'clustered': NEW_HOT[:4], 'scattered': [2, 6, 48, 52]}
    field = synthetic_sst()
    records = {}
    with TemporaryDirectory(prefix='sst-zarr-update-') as temp:
        temp = Path(temp)
        base_path = temp / 'base.zarr'
        create_array(base_path, BASE_CHUNKS)[:] = field
        reader = CountingStore(base_path)
        base = zarr.open_array(store=reader, mode='r', zarr_format=2)
        for name, added in cases.items():
            path = temp / f'{name}.zarr'
            target = create_array(path, SERIES_CHUNKS)
            build_replica(base, target, HOT)
            before = reader.payload_bytes
            build_replica(base, target, added)
            observed_read = reader.payload_bytes - before
            added = frozenset(map(int, added))
            evicted = set(map(int, HOT[4:]))
            observed_write = 0
            for tile in added | evicted:
                y, x = divmod(tile, TILE_SIDE)
                for t in range(SHAPE[0] // SERIES_CHUNKS[0]):
                    key = path / f'{t}.{y}.{x}'
                    if tile in evicted:
                        key.unlink()
                    else:
                        observed_write += key.stat().st_size
            assert (observed_read, observed_write) == update_cost(added)
            assert sum(f.stat().st_size for f in path.iterdir() if not f.name.startswith('.')) == BUDGET * REPLICA_POINT
            store = CountingStore(path)
            new_layout = zarr.open_array(store=store, mode='r', zarr_format=2)
            residents = set(map(int, HOT[:4])) | added
            for tile in residents | evicted:
                y, x = divmod(tile, TILE_SIDE)
                index = slice(None), y * TILE_SIDE, x * TILE_SIDE
                if tile in residents:
                    check_query(new_layout, store, field, index, chunk_cost(SERIES_CHUNKS, point=True))
                else:
                    check_query(base, reader, field, index, chunk_cost(BASE_CHUNKS, point=True))
            records[name] = dict(added_tiles=sorted(added), evicted_tiles=sorted(evicted),
                                 build_read_bytes=observed_read, build_write_bytes=observed_write,
                                 verified_point_queries=len(residents | evicted))
    return records


def net_percent(trace):
    return 100 * trace[-1]['cumulative_net_saved_bytes'] / sum(r['baseline_bytes'] for r in trace)


def paired_interval(delta):
    rng = np.random.default_rng(80001)
    draws = rng.choice(delta, (4000, len(delta)), replace=True).mean(axis=1)
    return np.quantile(draws, (.025, .975)).tolist()


def run_workloads():
    traces = {}
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for seed in SEEDS:
            initial, windows = make_trace(seed, scenario)
            for policy in POLICIES:
                traces[scenario, policy, seed] = replay(seed, scenario, initial, windows, policy)
        rolling = np.array([net_percent(traces[scenario, 'Rolling', seed]) for seed in SEEDS])
        for policy in POLICIES:
            current = [traces[scenario, policy, seed] for seed in SEEDS]
            rates = np.array([net_percent(r) for r in current])
            updates = np.mean([sum(row['tiles_added'] > 0 for row in r[1:]) for r in current])
            paired = rates - rolling
            summary[scenario][policy] = dict(mean_net_saved_pct=float(rates.mean()),
                                             mean_rebuild_windows=float(updates),
                                             paired_delta_vs_rolling_pp=float(paired.mean()),
                                             paired_bootstrap_95_pp=paired_interval(paired))
    return traces, summary


def sweep_burst_duration():
    rows = []
    for points, length in product((8, 16, 32, 64, 128), (1, 2, 3, 4, 6, 9, 12, 16)):
        for seed in SEEDS:
            initial, windows = make_trace(seed, 'Burst duration sweep', points, length)
            fixed, rolling, guarded = [net_percent(replay(seed, 'Burst duration sweep', initial, windows, p))
                                       for p in ('Fixed', 'Rolling', 'Guarded')]
            rows.append(dict(seed=seed, point_queries_per_window=points, burst_windows=length,
                             fixed_net_pct=fixed, rolling_net_pct=rolling, guarded_net_pct=guarded))
    return rows


def plot_windows(traces, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.1), sharex=True, sharey=True, layout='constrained')
    for ax, scenario in zip(axes.flat, SCENARIOS[:4]):
        for policy in ('Fixed', 'Rolling', 'Guarded'):
            series = np.array([[r['cumulative_net_saved_bytes'] for r in traces[scenario, policy, s]] for s in SEEDS])
            ax.plot(np.arange(1, WINDOWS + 1), series.mean(axis=0) / MIB,
                    label=policy, color=COLORS[policy], linewidth=1.9)
        if scenario == 'Persistent shift':
            ax.axvline(CHANGE + 1, color='.55', ls=':', linewidth=1)
        if scenario == 'One-window bursts':
            for t in (7, 13, 19, 25, 31):
                ax.axvline(t, color='.75', ls=':', linewidth=.8)
        ax.set_title(scenario, loc='left', fontsize=10)
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, ncols=3, fontsize=8, loc='upper left')
    fig.supxlabel('Completed query windows', fontsize=10)
    fig.supylabel('Cumulative net bytes saved (MiB)', fontsize=10)
    fig.savefig(out / 'adaptation_timeline.png', dpi=230)
    plt.close(fig)


def plot_sweep(rows, out):
    lengths = sorted({r['burst_windows'] for r in rows})
    rates = sorted({r['point_queries_per_window'] for r in rows})
    matrix = np.array([[np.mean([r['guarded_net_pct'] - r['rolling_net_pct'] for r in rows
                                if r['burst_windows'] == n and r['point_queries_per_window'] == q])
                        for n in lengths] for q in rates])
    extent = np.max(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(8.6, 3.65), layout='constrained')
    im = ax.imshow(matrix, cmap='BrBG', vmin=-extent, vmax=extent, aspect='auto')
    for y, x in product(range(len(rates)), range(len(lengths))):
        value = matrix[y, x]
        ax.text(x, y, f'{value:+.1f}', ha='center', va='center', fontsize=9,
                color='white' if abs(value) > extent * .57 else '#222222')
    ax.set(xticks=range(len(lengths)), xticklabels=lengths, yticks=range(len(rates)), yticklabels=rates,
           xlabel='Duration of the new hotspot (windows)', ylabel='Point queries per window')
    fig.colorbar(im, ax=ax, label='Guarded minus rolling (pp of baseline bytes)')
    fig.savefig(out / 'duration_sensitivity.png', dpi=230)
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    traces, summary = run_workloads()
    sensitivity = sweep_burst_duration()
    physical = incremental_audit()
    write_csv(out / 'adaptive_windows.csv', [row for trace in traces.values() for row in trace])
    write_csv(out / 'duration_sensitivity.csv', sensitivity)
    (out / 'adaptive_summary.json').write_text(json.dumps(
        dict(seed_range=[SEEDS[0], SEEDS[-1]], results=summary, incremental_zarr=physical),
        indent=2) + '\n', encoding='utf-8')
    plot_windows(traces, out)
    plot_sweep(sensitivity, out)
    for scenario, policies in summary.items():
        print(scenario, ' '.join(f"{p}={v['mean_net_saved_pct']:.2f}%" for p, v in policies.items()))


if __name__ == '__main__':
    main()
