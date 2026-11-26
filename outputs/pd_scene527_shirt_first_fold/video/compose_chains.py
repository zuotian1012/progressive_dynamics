import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

out = Path(__file__).resolve().parent
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
groups = [
    ('chain_4090', [('L3_start/L3','4,090 vertices'), ('L3_start/L4','7,021 vertices'),
                    ('L3_start/L5','13,974 vertices'), ('L3_start/L6','27,811 vertices')]),
    ('chain_7021', [('L4_start/L4','7,021 vertices'), ('L4_start/L5','13,974 vertices'), ('L4_start/L6','27,811 vertices')]),
    ('fine_55068_vs_native', [('L3_start/L6','PD++ from 4,090'), ('L4_start/L6','PD++ from 7,021'), ('direct/L6','Native IPC')]),
]
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
for title, cases in groups:
    args = [ffmpeg, '-hide_banner', '-loglevel','error','-y','-filter_complex_threads','3']
    filters = []
    four = len(cases) == 4
    width, height = (960,540) if four else (768,536)
    cols = 2 if four else 3
    pane_height = height + 42
    overlay = Image.new('RGBA',(cols*width,(2 if four else 1)*pane_height))
    draw = ImageDraw.Draw(overlay)
    for i,(case,label) in enumerate(cases):
        stem = case.split('/')[1] if case in ('L4_start/L4','L4_start/L5') else case.replace('/','_')
        metadata = json.loads((out/f'{stem}_oblique.replay.json').read_text())
        assert [metadata[k] for k in ('source_frame_start','source_frame_end','frames','record_fps')] == [0,800,401,30]
        args += ['-i',str(out/f'{stem}_oblique.mp4')]
        resize = 'crop=960:540:0:80' if four else 'crop=960:670:0:25,scale=768:536'
        filters.append(f'[{i}:v]{resize},pad={width}:{pane_height}:0:42:color=0x192432[p{i}]')
        origin = ((i%cols + 0.5)*width, (i//cols)*pane_height + 21)
        draw.text(origin,label,anchor='mm',font=font,fill='white')
    layout = '|'.join(f'{(i%cols)*width}_{(i//cols)*pane_height}' for i in range(len(cases)))
    filters.append(''.join(f'[p{i}]' for i in range(len(cases))) + f'xstack=inputs={len(cases)}:layout={layout}[grid]')
    filters.append(f'[grid][{len(cases)}:v]overlay=0:0[v]')
    with TemporaryDirectory() as tmp:
        header = Path(tmp)/'header.png'
        overlay.save(header)
        movie = out/f'{title}.mp4'
        subprocess.run(args + ['-i',str(header),'-filter_complex',';'.join(filters),'-map','[v]','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],check=True)
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(movie))
        assert count == 401 and abs(seconds-401/30)<0.03
        print(movie, count, seconds, flush=True)
