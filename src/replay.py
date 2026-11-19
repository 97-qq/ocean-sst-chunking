"""Replay synthetic SST queries and verify the byte model against local Zarr reads."""

import csv
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zarr
from zarr.storage import LocalStore


ROOT = Path(__file__).resolve().parents[1]
SHAPE = (240, 64, 64)
BASE_CHUNKS = (4, 32, 32)
SERIES_CHUNKS = (120, 8, 8)
TILES = 64
TILE_SIDE = 8
BUDGETS = (0, 4, 8, 16, 32, 64)
SEEDS = range(4101, 4113)
SCENARIOS = ('Stable', 'Shifted', 'Uniform')
HOT = np.array([0, 1, 8, 9, 16, 17, 24, 25])
NEW_HOT = np.array([38, 39, 46, 47, 54, 55, 62, 63])
DTYPE = np.dtype('<f4')
MIB = 1024**2


class CountingStore(LocalStore):
    def __init__(self, root):
        super().__init__(root, read_only=True)
        self.chunks = 0
        self.payload_bytes = 0

    async def get(self, key, prototype=None, byte_range=None):
        buffer = await super().get(key, prototype, byte_range)
        if buffer is not None and not key.startswith('.'):
            self.chunks += 1
            self.payload_bytes += len(buffer)
        return buffer


def chunk_cost(chunks, *, point):
    count = SHAPE[0] // chunks[0] if point else (
        SHAPE[1] // chunks[1] * (SHAPE[2] // chunks[2])
    )
    return count, count * int(np.prod(chunks)) * DTYPE.itemsize


def draw_tiles(rng, count, hotspots):
    tiles = rng.integers(TILES, size=count)
    if hotspots is not None:
        use_hotspot = rng.random(count) < 0.85
        tiles[use_hotspot] = rng.choice(hotspots, size=use_hotspot.sum())
    return tiles


def train_ranking(rng):
    frequencies = np.bincount(draw_tiles(rng, 400, HOT), minlength=TILES)
    return np.lexsort((np.arange(TILES), -frequencies))


def replay_models():
    base_point = chunk_cost(BASE_CHUNKS, point=True)[1]
    series_point = chunk_cost(SERIES_CHUNKS, point=True)[1]
    map_cost = chunk_cost(BASE_CHUNKS, point=False)[1]
    rows = []
    for seed in SEEDS:
        streams = np.random.SeedSequence(seed).spawn(4)
        ranking = train_ranking(np.random.default_rng(streams[0]))
        for scenario, stream, hotspots in zip(SCENARIOS, streams[1:], (HOT, NEW_HOT, None)):
            points = draw_tiles(np.random.default_rng(stream), 640, hotspots)
            baseline = 640 * base_point + 160 * map_cost
            maximum_savings = 640 * (base_point - series_point)
            for budget in BUDGETS:
                hits = int(np.count_nonzero(np.isin(points, ranking[:budget])))
                saved = hits * (base_point - series_point)
                rows.append(dict(seed=seed, scenario=scenario, tiles=budget,
                                 point_hits=hits, point_hit_rate=hits / 640,
                                 query_payload_bytes=baseline - saved,
                                 mean_query_mib=(baseline - saved) / (800 * MIB),
                                 read_savings_pct=100 * saved / baseline,
                                 full_savings_retained_pct=100 * saved / maximum_savings))
    return rows


def synthetic_sst():
    rng = np.random.default_rng(2061)
    month = np.arange(SHAPE[0])[:, None, None]
    lat = np.linspace(25, 40, SHAPE[1])[None, :, None]
    lon = np.linspace(125, 145, SHAPE[2])[None, None, :]
    field = (23 - 0.30 * (lat - 25) + 3 * np.sin(2 * np.pi * month / 12)
             + 0.5 * np.cos((lon - 125) / 3) + rng.normal(0, 0.25, SHAPE)).astype(DTYPE)
    field[:, 0, 0] = np.nan  # Verify that a missing series survives either layout.
    return field


def create_array(path, chunks):
    return zarr.create_array(store=path, shape=SHAPE, chunks=chunks, dtype=DTYPE,
                             zarr_format=2, compressors=None, fill_value=np.nan,
                             config={'write_empty_chunks': True})


def spatial_parent(tile):
    tiles_per_parent = BASE_CHUNKS[1] // TILE_SIDE
    y, x = divmod(int(tile), TILE_SIDE)
    return y // tiles_per_parent, x // tiles_per_parent


def build_replica(base, target, tiles):
    parents = {}
    for tile in tiles:
        y, x = divmod(int(tile), TILE_SIDE)
        parents.setdefault(spatial_parent(tile), []).append((y, x))
    for (py, px), children in sorted(parents.items()):
        y0, x0 = py * BASE_CHUNKS[1], px * BASE_CHUNKS[2]
        slab = base[:, y0:y0 + BASE_CHUNKS[1], x0:x0 + BASE_CHUNKS[2]]
        for y, x in children:
            row, col = y * TILE_SIDE, x * TILE_SIDE
            target[:, row:row + TILE_SIDE, col:col + TILE_SIDE] = slab[
                :, row - y0:row - y0 + TILE_SIDE, col - x0:col - x0 + TILE_SIDE
            ]


def check_query(array, store, field, index, expected):
    before = (store.chunks, store.payload_bytes)
    actual = array[index]
    np.testing.assert_array_equal(actual, field[index])
    observed = (store.chunks - before[0], store.payload_bytes - before[1])
    if observed != expected:
        raise AssertionError(f'Chunk cost mismatch at {index}: {observed} vs {expected}')


def physical_audit():
    field = synthetic_sst()
    ranking = train_ranking(np.random.default_rng(np.random.SeedSequence(SEEDS[0]).spawn(4)[0]))
    selected = set(map(int, ranking[:8]))
    with TemporaryDirectory(prefix='sst-zarr-') as work:
        work = Path(work)
        base_path = work / 'base.zarr'
        create_array(base_path, BASE_CHUNKS)[:] = field
        reader = CountingStore(base_path)
        base = zarr.open_array(store=reader, mode='r', zarr_format=2)
        stats = {}
        replicas = {}
        for name, tiles in [('partial', selected), ('full', range(TILES))]:
            path = work / f'{name}.zarr'
            before = reader.payload_bytes
            start = perf_counter()
            build_replica(base, create_array(path, SERIES_CHUNKS), tiles)
            build_seconds = perf_counter() - start
            payload = sum(f.stat().st_size for f in path.iterdir() if not f.name.startswith('.'))
            assert payload == len(tiles) * chunk_cost(SERIES_CHUNKS, point=True)[1]
            stats[name] = dict(data_bytes=payload,
                               build_read_bytes=reader.payload_bytes - before,
                               build_seconds=build_seconds)
            store = CountingStore(path)
            replicas[name] = zarr.open_array(store=store, mode='r', zarr_format=2), store

        checked = 0
        for mode in ('base', 'partial', 'full'):
            for tile in range(TILES):
                y, x = divmod(tile, TILE_SIDE)
                index = (slice(None), y * TILE_SIDE, x * TILE_SIDE)
                if mode == 'full' or (mode == 'partial' and tile in selected):
                    array, store = replicas[mode]
                    chunks = SERIES_CHUNKS
                else:
                    array, store = base, reader
                    chunks = BASE_CHUNKS
                check_query(array, store, field, index, chunk_cost(chunks, point=True))
                checked += 1
            for month in (0, 77, 239):
                index = (month, slice(None), slice(None))
                check_query(base, reader, field, index, chunk_cost(BASE_CHUNKS, point=False))
                checked += 1

    return {'shape': SHAPE, 'seed_range': [SEEDS[0], SEEDS[-1]],
            'base_chunks': BASE_CHUNKS, 'series_chunks': SERIES_CHUNKS,
            'base_data_bytes': field.nbytes, 'checked_queries': checked,
            'base_point_chunks_bytes': chunk_cost(BASE_CHUNKS, point=True),
            'series_point_chunks_bytes': chunk_cost(SERIES_CHUNKS, point=True),
            'map_chunks_bytes': chunk_cost(BASE_CHUNKS, point=False),
            'selected_tiles_seed_4101': sorted(selected), 'replicas': stats,
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'zarr': zarr.__version__, 'matplotlib': matplotlib.__version__}}


def values(rows, scenario, budget, key):
    return np.array([r[key] for r in rows if r['scenario'] == scenario and r['tiles'] == budget])


def plot_results(rows, out):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True})
    fig, ax = plt.subplots(figsize=(8, 3.65), layout='constrained')
    x = np.arange(len(SCENARIOS))
    for i, (budget, label, color) in enumerate(zip(
        (0, 8, 64), ('Original', 'Hot tiles (8/64)', 'Full replica'),
        ('#59677a', '#137b75', '#d08637'),
    )):
        samples = [values(rows, scenario, budget, 'mean_query_mib') for scenario in SCENARIOS]
        means = [s.mean() for s in samples]
        ax.bar(x + (i - 1) * .24, means, .22, yerr=[s.std(ddof=1) for s in samples],
               label=label, color=color, capsize=3, error_kw={'elinewidth': 1})
    ax.set(xticks=x, xticklabels=SCENARIOS, ylabel='Chunk payload (MiB/query)', ylim=(0, .92))
    ax.grid(axis='y', color='#e4e8eb')
    ax.legend(loc='upper center', ncols=3, frameon=False)
    fig.savefig(out / 'query_payload.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.65), layout='constrained')
    x = 100 * np.array(BUDGETS) / TILES
    for scenario, color, marker in zip(SCENARIOS, ('#137b75', '#c05a42', '#526881'), ('o', 's', '^')):
        samples = [values(rows, scenario, n, 'read_savings_pct') for n in BUDGETS]
        ax.plot(x, [s.mean() for s in samples], label=scenario, color=color, marker=marker)
    ax.axvline(12.5, linewidth=1, linestyle=':', color='#697380')
    ax.set(xlabel='Extra chunk storage (% of original)', ylabel='Read savings vs original (%)',
           xlim=(-2, 102), ylim=(-3, 100), xticks=[0, 12.5, 25, 50, 75, 100])
    ax.grid(color='#e4e8eb')
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(out / 'storage_tradeoff.png', dpi=220)
    plt.close(fig)


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    rows = replay_models()
    audit = physical_audit()
    with (out / 'replay.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'zarr_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    plot_results(rows, out)
    print(f"{len(rows)} replay rows and {audit['checked_queries']} verified Zarr queries")
    for scenario in SCENARIOS:
        savings = values(rows, scenario, 8, 'read_savings_pct')
        print(f'{scenario:<8} {savings.mean():.2f}% +/- {savings.std(ddof=1):.2f} pp')


if __name__ == '__main__':
    main()
