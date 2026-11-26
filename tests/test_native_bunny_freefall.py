"""Regression for contact-free drift in the native bunny solve."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.measure_bunny_freefall import measure
from scripts.bunny_runtime import audit_bunny_contact


ROOT = Path(__file__).resolve().parents[1]


class NativeBunnyFreefall(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'output/bunny_paper_assets/mesh_7.npz').is_file(), 'requires the reconstructed bunny hierarchy')
    def test_flat_cloth_has_no_initial_self_contact_at_point_seven_mm(self):
        from scripts.audit_shirt_initial_contact import audit
        from scripts.bunny_runtime import BUNNY_CONTACT_DISTANCE

        rows = audit(ROOT / 'output/bunny_paper_assets', dhat=BUNNY_CONTACT_DISTANCE, clearance=.00007)
        self.assertEqual([r['level'] for r in rows], list(range(8)))
        self.assertEqual([r['initial_active_contact_count'] for r in rows], [0] * 8)

    @unittest.skipUnless(os.environ.get('BUNNY_ORDERED_REFERENCE'), 'requires an audited zero-friction contact trajectory')
    def test_repeated_contact_step_has_identical_positions(self):
        import numpy as np
        from scripts.bunny_runtime import BUILD_ROOT

        reference=Path(os.environ['BUNNY_ORDERED_REFERENCE']).resolve()
        package=BUILD_ROOT/'bunny_ordered_fem_runtime/src'
        with tempfile.TemporaryDirectory() as tmp:
            positions=[]
            for name in ('first','repeat'):
                run=Path(tmp)/name
                result=subprocess.run(
                    [sys.executable,'scripts/run_bunny_ground_l0.py','--scene','bunny',
                     '--assets','output/bunny_paper_assets','--native-package',str(package),
                     '--output',str(run),'--restart',str(reference),'--restart-frame','39',
                     '--steps','1','--max-newton','1500','--friction','0','--frozen-offset',
                     '--refined-probe','--adaptive-search'],
                    cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode,0,result.stdout[-2000:].decode(errors='replace'))
                state=np.load(run/'positions.npy')
                self.assertGreater(np.linalg.norm(state[-1]-state[0]),0.)
                positions.append(state[-1])
            np.testing.assert_array_equal(*positions)

    @unittest.skipUnless(os.environ.get('BUNNY_CUDA_TEST') == '1', 'requires native bunny CUDA build and assets')
    def test_frozen_gradient_without_friction_or_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'frozen'
            result = subprocess.run(
                [sys.executable, 'scripts/run_bunny_ground_l0.py', '--scene', 'bunny',
                 '--assets', 'output/bunny_paper_assets', '--output', str(run), '--steps', '19',
                 '--friction', '0', '--frozen-offset', '--refined-probe', '--adaptive-search'],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:].decode(errors='replace'))
            meta = json.loads((run / 'run.json').read_text())
            self.assertEqual(meta['bunny_linear_solver'], 'eigen_sparse_ldlt')
            self.assertTrue(meta['ordered_fem_assembly'])
            self.assertTrue(meta['frozen_offset_only'])
            self.assertFalse(meta['exact_offset_gradient'] or meta['initial_non_descent_rescue'])
            self.assertIn(b'FLAG_PSQ_EXACT_OFFSET enabled=false', result.stdout)
            self.assertNotIn(b'FLAG_PSQ_RESCUE frame=', result.stdout)
            self.assertFalse(audit_bunny_contact(run, meta)['friction_systems_active'])

    @unittest.skipUnless(os.environ.get('BUNNY_CUDA_REFERENCE'), 'requires an audited zero-friction L0')
    def test_one_step_restarts_from_the_saved_contact_state(self):
        reference = Path(os.environ['BUNNY_CUDA_REFERENCE']).resolve()
        meta = json.loads((reference / 'run.json').read_text())
        if meta['contact_friction']['cloth_bunny'] != 0:
            self.skipTest('this control requires zero friction')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'one_step'
            with run.with_suffix('.log').open('wb') as log:
                subprocess.run(
                    [sys.executable, 'scripts/run_bunny_ground_l0.py', '--scene', 'bunny',
                     '--assets', meta['assets'], '--output', str(run), '--steps', '1',
                     '--restart', str(reference), '--restart-frame', '18',
                     '--max-newton', str(meta['config']['newton']['max_iter']),
                     '--native-package', meta['native_package'], '--exact-offset', '--refined-probe', '--adaptive-search'],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            restart = json.loads((run / 'run.json').read_text())
            self.assertEqual(restart['completed_steps'], 1)
            self.assertTrue(restart['diagnostic_restart']['initial_state_readback_verified'])
            self.assertEqual(restart['contact_friction'], meta['contact_friction'])

    @unittest.skipUnless(os.environ.get('BUNNY_CUDA_RMS_TEST') == '1', 'requires the bunny RMS CUDA build')
    def test_rms_convergence_history_before_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'rms'
            with run.with_suffix('.log').open('wb') as output:
                subprocess.run(
                    [sys.executable, 'scripts/run_bunny_ground_l0.py', '--scene', 'bunny',
                     '--assets', 'output/bunny_paper_assets', '--output', str(run), '--steps', '13',
                     '--native-package', 'output/flag_L3_fix_20261118/bunny_rms_runtime/src',
                     '--exact-offset', '--refined-probe', '--adaptive-search'],
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, 'scripts/audit_flag_fine_search.py', str(run),
                            '--check-converged-energy'], cwd=ROOT, check=True, stdout=subprocess.PIPE)
            audit = json.loads((run / 'line_search_audit.json').read_text())
            self.assertEqual(audit['converged_energy_checks'], 13)
            self.assertGreaterEqual(audit['rms_decrements_checked'], 3 * 13)
            self.assertLess(max(row['max_mm'] for row in measure(run)['frames']), .07)

    @unittest.skipUnless(os.environ.get('BUNNY_CUDA_TEST') == '1', 'requires native bunny CUDA build and assets')
    def test_thirteen_steps_before_first_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'L0'
            result = subprocess.run(
                [sys.executable, 'scripts/run_bunny_ground_l0.py', '--scene', 'bunny',
                 '--assets', 'output/bunny_paper_assets', '--output', str(run), '--steps', '13',
                 '--exact-offset', '--refined-probe', '--adaptive-search'],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:].decode(errors='replace'))
            contact = audit_bunny_contact(run, json.loads((run / 'run.json').read_text()))
            self.assertEqual(contact['cloth_bunny'], 0.3)
            frames = measure(run)['frames']
            self.assertEqual(len(frames), 13)
            # Even the worst vertex should stay within one physical cloth thickness.
            self.assertLess(max(row['max_mm'] for row in frames), 0.07)

    @unittest.skipUnless(os.environ.get('BUNNY_CUDA_REFERENCE'), 'requires an audited four-second bunny L0')
    def test_both_fine_integrators_in_freefall(self):
        reference = Path(os.environ['BUNNY_CUDA_REFERENCE']).resolve()
        parent = json.loads((reference / 'run.json').read_text())
        self.assertEqual(parent['level'], 0)
        self.assertTrue(parent['exact_offset_gradient'])
        self.assertTrue(json.loads((reference / 'continuation_audit.json').read_text())['passed'])
        thickness_mm = 1000 * parent['material']['thickness']
        with tempfile.TemporaryDirectory() as tmp:
            for method in ('semi', 'velpro'):
                with self.subTest(method=method):
                    run = Path(tmp) / method
                    result = subprocess.run(
                        [sys.executable, 'scripts/run_bunny_ground_fine.py', '--scene', 'bunny',
                         '--level', '1', '--method', method, '--lower', str(reference),
                         '--assets', parent['assets'], '--output', str(run), '--steps', '13',
                         '--exact-offset', '--refined-probe', '--adaptive-search'],
                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    self.assertEqual(result.returncode, 0, result.stdout[-2000:].decode(errors='replace'))
                    contact = audit_bunny_contact(run, json.loads((run / 'run.json').read_text()))
                    self.assertEqual(contact['cloth_bunny'], parent['contact_friction']['cloth_bunny'])
                    frames = measure(run)['frames']
                    self.assertEqual(len(frames), 13)
                    self.assertLess(max(row['max_mm'] for row in frames), thickness_mm)
