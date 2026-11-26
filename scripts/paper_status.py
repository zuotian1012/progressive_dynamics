"""Read compact live status, tolerating a checkpoint writer's short file update."""
import argparse
import json
from pathlib import Path
import time

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
for branch in ['flag_budget1000','bunny_event_window']:
    for path in sorted((a.root/branch).glob('L*/run.json')):
        for attempt in range(4):
            try:m=json.loads(path.read_text());break
            except (OSError,ValueError):
                m=None;time.sleep(.02)
        if m is None:print(f'{branch}/{path.parent.name}: checkpoint update in progress');continue
        print(f'{branch}/{path.parent.name}: {m["status"]}, {m["completed_steps"]}/{m["steps"]} steps, {m["vertices"]:,} vertices')
