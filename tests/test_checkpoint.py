"""Regression for the real checkpoint-rest-geometry replay error."""
import unittest
from pathlib import Path
import subprocess
import sys
import tempfile
import json
import numpy as np
import ipctk
from progressive.runtime import numerical_threads


class CheckpointTests(unittest.TestCase):
    def test_thread_budget_is_restored(self):
        before=ipctk.get_num_threads()
        with numerical_threads(1) as runtime:
            self.assertEqual(ipctk.get_num_threads(),1)
            self.assertTrue(all(p['num_threads']==1 for p in runtime['blas']))
        self.assertEqual(ipctk.get_num_threads(),before)

    @unittest.skipUnless(Path('output/pd_shirt_intrinsic_sequence/velpro_L1/run.json').exists(),
                         'Requires the saved Scene527 regression checkpoint')
    def test_replay_keeps_original_collision_rest(self):
        from scripts.replay_progressive_step import load_replay
        with numerical_threads(1):
            problem,q,args,digests=load_replay(Path('output/pd_shirt_intrinsic_sequence'),1,21)
            reference=problem.embedding.positions(problem.embedding.q0)
            np.testing.assert_array_equal(problem.contact.mesh.rest_positions,reference)
            self.assertGreater(np.linalg.norm(problem.world(q)-reference),.001)
            self.assertNotEqual(digests['state_sha256'],digests['collision_rest_sha256'])
            np.testing.assert_array_equal(problem.previous,q)

    def test_resumed_friction_trajectory_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def execute(name,steps,resume=None,interrupted=False):
                cmd=[sys.executable,'-m','progressive.run','--scene','grip','--base-segments','2',
                     '--levels','2','--steps',str(steps),'--methods','velpro','direct','--threads','1',
                     '--output',str(root/name)]
                if resume:cmd+=['--resume-from',str(root/resume)]
                if interrupted:cmd+=['--resume-interrupted']
                subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
            execute('prefix',4);execute('resumed',8,'prefix');execute('whole',8)
            for method in ('velpro','direct'):
                for level in (0,1):
                    for name in ('positions.npy','robot.npy'):
                        a=np.load(root/'resumed'/f'{method}_L{level}'/name)
                        b=np.load(root/'whole'/f'{method}_L{level}'/name)
                        np.testing.assert_array_equal(a,b)
            cfg=json.loads((root/'prefix'/'experiment.json').read_text())
            cfg.update(completed=False,runs=[])
            (root/'prefix'/'experiment.json').write_text(json.dumps(cfg))
            for meta_path in (root/'prefix').glob('*_L*/run.json'):
                meta=json.loads(meta_path.read_text());meta['completed']=False
                meta_path.write_text(json.dumps(meta))
            execute('recovered',8,'prefix',True)
            for method in ('velpro','direct'):
                for level in (0,1):
                    np.testing.assert_array_equal(np.load(root/'recovered'/f'{method}_L{level}'/'positions.npy'),
                                                  np.load(root/'whole'/f'{method}_L{level}'/'positions.npy'))


if __name__=='__main__':unittest.main()
