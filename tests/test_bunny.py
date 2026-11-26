"""Bunny integration: free cloth, fixed collider, and reproducible inputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import trimesh
from progressive.run import main
from progressive.checkpoint import validate_resume
from progressive.audit_bunny import audit


class BunnyTests(unittest.TestCase):
    def test_config_override_and_fixed_obstacle_freefall(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assets=root/'assets'; assets.mkdir()
            x=np.array([[-.1,-.1,1.],[.1,-.1,1.],[.1,.1,1.],[-.1,.1,1.]])
            f=np.array([[0,1,2],[0,2,3]])
            obstacle=trimesh.creation.icosphere(subdivisions=0,radius=.1)
            np.savez(assets/'mesh_0.npz',vertices=x,faces=f)
            np.savez(assets/'bunny.npz',vertices=obstacle.vertices,faces=obstacle.faces)
            config=root/'config.json'
            config.write_text(json.dumps({'parameters':{'scene':'bunny','assets':str(assets),
                'levels':1,'steps':4,'dt':.01,'methods':['velpro'],'poisson':.23,'threads':1}}))
            output=root/'run'
            main(['--config',str(config),'--steps','2','--output',str(output)])
            saved=json.loads((output/'experiment.json').read_text())
            self.assertTrue(saved['completed']); self.assertEqual(saved['steps'],2)
            self.assertEqual(saved['material']['poisson'],.23)
            track=np.load(output/'velpro_L0'/'positions.npy')
            # Backward Euler free fall, no accidental pins or moving bunny DOFs.
            np.testing.assert_allclose(track[-1],x+[0,0,-3*9.81*.01**2],atol=1e-10)
            self.assertEqual(np.load(output/'velpro_L0'/'robot.npy').shape,(3,0,4,4))
            with np.load(output/'velpro_L0'/'fixed_collider_0.npz') as fixed:
                np.testing.assert_array_equal(fixed['vertices'],obstacle.vertices)
            validate_resume(output,saved)
            changed=copy.deepcopy(saved)
            changed['asset_files_sha256']['bunny.npz']='changed'
            with self.assertRaisesRegex(ValueError,'geometry or intrinsic-map'):
                validate_resume(output,changed)
            self.assertEqual(audit(output)['runs'][0]['geometry_frames_checked'],3)
            # A failed/incomplete run can contain arbitrary preallocated tail data.
            saved['completed']=False
            (output/'experiment.json').write_text(json.dumps(saved))
            folder=output/'velpro_L0'
            meta=json.loads((folder/'run.json').read_text())
            meta.update(completed=False,completed_steps=1)
            (folder/'run.json').write_text(json.dumps(meta))
            states=np.load(folder/'positions.npy',mmap_mode='r+')
            states[2]=np.nan;states.flush()
            self.assertEqual(audit(output,accepted_prefix=True)['runs'][0]['geometry_frames_checked'],2)
            del states
            changed=copy.deepcopy(saved)
            changed['asset_files_sha256']['map_0_1.npz']='changed'
            with self.assertRaisesRegex(ValueError,'geometry or intrinsic-map'):
                validate_resume(output,changed)

    def test_invalid_config_method_is_rejected_before_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'invalid.json'
            config.write_text(json.dumps({'parameters':{'methods':['not-an-integrator']}}))
            with patch('progressive.run.run') as run, self.assertRaises(SystemExit):
                main(['--config',str(config),'--output',str(Path(directory)/'run')])
            run.assert_not_called()


if __name__=='__main__':unittest.main()
