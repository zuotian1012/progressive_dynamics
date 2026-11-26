"""Synchronize four completed shirt folds by physical step in a 2x2 video."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import imageio.v2 as imageio
import numpy as np
from matplotlib import font_manager
from shirt_demo.report import load_run, draw, plt

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / 'output/shirt_four_level_full_fold_20261118/L0/run',
    ROOT / 'output/shirt_four_level_full_fold_20261118/L1/run',
    ROOT / 'output/shirt_L2_full_fold_mas_20261119/full_protocol/run',
    ROOT / 'output/shirt_L3_solver_fix_20261119/full_fold/run',
]
LABELS = ['diag', 'diag', '修复后的 MAS', '补偿 PCG + 修复后的 MAS']
PHASES = [(60, '沉降'), (212, '夹爪接近'), (356, '夹取定位'),
          (596, '闭合抓取'), (656, '夹持保持'), (896, '抬升'),
          (1076, '抬升保持'), (1436, '平移'), (1616, '平移保持'),
          (2216, '翻折'), (2576, '放下'), (2696, '放置保持'),
          (2936, '张爪释放'), (3116, '夹爪撤离'), (3476, '静置观察')]
DATA = None


def initialize(paths, bounds):
    global DATA
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    DATA = [load_run(p) for p in paths]
    for data in DATA:
        data['plot_bounds'] = bounds


def render(step):
    phase = next(label for end, label in PHASES if step <= end)
    fig = plt.figure(figsize=(14.4, 10.8), dpi=100, facecolor='white')
    fig.subplots_adjust(left=.025, right=.975, bottom=.075, top=.88, wspace=.015, hspace=.16)
    fig.text(.5, .966, '四层衣服仿真 · 完整一折', ha='center', fontsize=23, weight='bold')
    fig.text(.5, .93, f'{phase}  |  仿真时间 {step / 120:05.2f} / {3476 / 120:.2f} 秒  |  第 {step} 步',
             ha='center', fontsize=14, color='#34485b')
    for i, data in enumerate(DATA):
        meta = data['meta']
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        draw(ax, data, step)
        ax.view_init(elev=35, azim=-70)
        ax.set_title(f'L{i}   {meta["vertices"]:,} 顶点 / {meta["faces"]:,} 三角面\n{LABELS[i]}',
                     fontsize=12, pad=4)
        ax.plot([1.15, 1.15], [-.50, .50], [.8, .8], color='#b35e20', linewidth=1.7)
        ax.tick_params(labelsize=7, pad=0)
        ax.set_xlabel('x (m)', labelpad=0, fontsize=8)
        ax.set_ylabel('y (m)', labelpad=0, fontsize=8)
        ax.set_zlabel('z (m)', labelpad=0, fontsize=8)
    fig.text(.5, .035, '同一仿真时刻 · 约 1× 回放 · 统一视角与尺度 · 橙线为桌面远端边界',
             ha='center', fontsize=11, color='#34485b')
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/shirt_four_level_full_fold_comparison_20261119')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((p / 'run.json').read_text()) for p in SOURCES]
    assert all(m['completed'] and m['completed_steps'] == m['steps'] == 3476 for m in metas)
    assert [m['level'] for m in metas] == list(range(4))
    assert len({m['dt'] for m in metas}) == len({m['robot_q_sha256'] for m in metas}) == 1
    assert metas[0]['dt'] == 1 / 120
    for i in range(1, 4):
        assert Path(metas[i]['lower_run']).resolve() == SOURCES[i - 1].resolve()
        assert metas[i]['lower_run_sha256'] == hashlib.sha256((SOURCES[i - 1] / 'run.json').read_bytes()).hexdigest()
    # One saved state every eight physics steps, plus the exact final state.
    # No simulation state is interpolated or changed.
    steps = np.unique(np.r_[np.arange(0, 3477, 8), 3476]).astype(int)
    low = .60
    for p in SOURCES:
        positions = np.load(p / 'positions.npy', mmap_mode='r')
        low = min(low, min(float(positions[k, :, 2].min()) for k in steps) - .02)
    bounds = ((.15, 1.30), (-.55, .55), (low, 1.28))
    manifest = dict(status='preview' if args.preview_only else 'rendering',
        sources=[dict(path=str(p), level=m['level'], vertices=m['vertices'], faces=m['faces'],
                      physics_seconds=m['physics_seconds'], solver_label=LABELS[i],
                      convergence_policy=m['convergence_policy'], metadata_sha256=hashlib.sha256((p / 'run.json').read_bytes()).hexdigest())
                 for i, (p, m) in enumerate(zip(SOURCES, metas))],
        same_trajectory=True, continuous_lower_level_chain_verified=True,
        simulation_seconds=3476 / 120, fps=15, frames=len(steps),
        video_seconds=len(steps) / 15, sample_steps=steps.tolist(), bounds=bounds,
        note='Latest completed run at each level; solver and convergence policies differ as recorded. All four panels use the exact same saved physical step. Rendering only; no resimulation, smoothing or state interpolation.')
    path = args.output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    initialize(SOURCES, bounds)
    imageio.imwrite(args.output / 'final_comparison.png', render(3476))
    if args.preview_only:
        imageio.imwrite(args.output / 'grasp_preview.png', render(596))
        print('Previews ready', flush=True)
        return
    video = args.output / 'four_levels_full_fold.mp4'
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(SOURCES, bounds)) as pool:
        with imageio.get_writer(video, fps=15, codec='libx264', quality=8, macro_block_size=2,
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for count, frame in enumerate(pool.map(render, map(int, steps), chunksize=1), 1):
                writer.append_data(frame)
                if count % 25 == 0 or count == len(steps):
                    print(f'Rendered {count}/{len(steps)} frames', flush=True)
    manifest['status'] = 'completed'
    manifest['video_sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Completed', video, flush=True)


if __name__ == '__main__':
    main()
