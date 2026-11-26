import json
from pathlib import Path
import unittest
import tempfile
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from scipy.sparse import coo_matrix
from cloth.mesh import grid
from shirt_demo.assets import read_obj
from progressive.expansion_history import ExpansionHistory
from progressive.prolongation import ShellProlongation
from progressive.safe_expansion import initialize
from progressive.refinement_contact import (ExpansionOptions,RefinementFailure,
    collision_mesh,minimum_gap,safe_perturb,ratio_advance,rank_step)
from progressive.refinement_relaxation import arap_local,strain_relaxation,correction_energy
from progressive.shell import Shell,local_hessian


def fixture(name='intrinsic_expansion'):
    root=Path(__file__).parent/'fixtures'/name
    coarse,cf=read_obj(root/'mesh_1.obj');fine,ff=read_obj(root/'mesh_0.obj')
    record=json.loads((root/'expansion_1.json').read_text())
    h=ExpansionHistory(record,coarse,cf,fine,ff)
    data=np.loadtxt(root/'map_1.txt',skiprows=1)
    U=coo_matrix((data[:,2],(data[:,0].astype(int),data[:,1].astype(int))),shape=(len(fine),len(coarse))).tocsr()
    P=ShellProlongation(coarse,fine,cf,U,np.loadtxt(root/'anchors_1.txt',dtype=int))
    return h,P,record


class SafeExpansionTests(unittest.TestCase):
    def test_split_endpoints_are_new_even_when_the_parent_slot_is_reused(self):
        h,P,_=fixture('intrinsic_grip')
        np.testing.assert_array_equal(h.unsplit_coarse_ids,[24])
        with np.load(Path(__file__).parent/'fixtures/intrinsic_grip/failed_step_input.npz') as saved:
            a={k:saved[k] for k in saved.files}
        target=P.position(a['coarse']);obstacles=[(a['obstacle_vertices'],a['obstacle_faces'])]
        with self.assertRaisesRegex(RefinementFailure,'Fixed inherited endpoints'):
            initialize(h,a['coarse'],target,obstacles=obstacles,contact_stiffness=1e4,
                       options=ExpansionOptions(minimum_gap=.00021,fixed_policy='coarse-indices'))
        from progressive.safe_expansion import strain_relaxation as original
        observed={}
        def check_fixed(shell,x,target,mask,*args):
            observed['mask']=mask.copy();observed['before']=x.copy()
            out,records=original(shell,x,target,mask,*args)
            np.testing.assert_array_equal(out[mask],x[mask])
            observed['after']=out.copy()
            return out,records
        with patch('progressive.safe_expansion.strain_relaxation',check_fixed):
            x,r=initialize(h,a['coarse'],target,obstacles=obstacles,contact_stiffness=1e4,
                           options=ExpansionOptions(minimum_gap=.00021))
        self.assertTrue(r['accepted']);self.assertLess(r['max_stretch'],1.1)
        self.assertGreater(r['final_gap_lower_bound'],.0002)
        self.assertEqual(r['strain_fixed_cloth_ids'],[24])
        self.assertTrue(observed['mask'][h.n:].all())
        self.assertGreater(np.linalg.norm(observed['after'][6]-observed['before'][6]),1e-4)

    def test_expansion_epsilon_is_distinct_and_cannot_weaken_thickness(self):
        h,P,_=fixture()
        with self.assertRaisesRegex(ValueError,'strictly exceed'):
            initialize(h,h.coarse_rest,P.position(h.coarse_rest),
                       options=ExpansionOptions(minimum_gap=.0002),dmin=.0002)
        x,r=initialize(h,h.coarse_rest,P.position(h.coarse_rest),
                       options=ExpansionOptions(minimum_gap=.00021))
        self.assertTrue(r['accepted'])
        self.assertTrue(all(b['required_gap']==.00021 for b in r['batches']))
        self.assertGreater(r['batches'][0]['gap_before'],.00021)
        np.testing.assert_allclose(x,h.fine_rest,atol=1e-12,rtol=0)

    def test_safe_seed_converges_to_same_fine_incremental_solution(self):
        from scripts.validate_refinement_dynamics import compare_seeds
        h,P,record=fixture()
        plane=np.array([[-2.,-1.,-.2],[2.,-1.,-.2],[2.,2.,-.2],[-2.,2.,-.2]])
        pf=np.array([[0,1,2],[0,2,3]]);nf,nc=len(h.fine_rest),len(h.coarse_rest)
        record['fine_vertex_count']+=4;record['coarse_ids']+=list(range(nf,nf+4))
        h=ExpansionHistory(record,np.vstack((h.coarse_rest,plane)),np.vstack((h.coarse_faces,pf+nc)),
                           np.vstack((h.fine_rest,plane)),np.vstack((h.fine_faces,pf+nf)))
        coarse=h.coarse_rest.copy();coarse[:nc,2]=0;coarse[nc:,2]=-.002
        naive=np.vstack((P.position(coarse[:nc]),coarse[nc:]))
        refined,_=initialize(h,coarse,naive)
        previous=h.fine_rest.copy();previous[:nf,2]=0;previous[nf:,2]=-.002
        states,r,s=compare_seeds(h.fine_rest,h.fine_faces,previous,refined)
        self.assertTrue(s['both_converged']);self.assertTrue(s['same_final_stiffness'])
        self.assertLess(s['final_solution_difference_rms_m'],1e-5)
        self.assertLess(abs(s['final_energy_difference']),1e-7)
        self.assertTrue(all(v['gap_lower_bound']>.0002 for v in r.values()))

    def test_vertical_resume_matches_continuous_and_checks_history(self):
        from scipy.sparse import save_npz
        h,P,record=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);assets=root/'assets';assets.mkdir()
            np.savez(assets/'mesh_0.npz',vertices=h.coarse_rest,faces=h.coarse_faces)
            np.savez(assets/'mesh_1.npz',vertices=h.fine_rest,faces=h.fine_faces)
            save_npz(assets/'map_0_1.npz',P.U);np.save(assets/'anchors_0_1.npy',P.anchors)
            history=assets/'expansion_0_1.json';history.write_text(json.dumps(record))
            def execute(name,steps,resume=None,fixed_policy=None):
                cmd=[sys.executable,'-m','progressive.run','--scene','intrinsic','--assets',str(assets),
                     '--steps',str(steps),'--levels','2','--methods','velpro','--warm-start','vertical',
                     '--output',str(root/name)]
                if resume:cmd+=['--resume-from',str(root/resume)]
                if fixed_policy:cmd+=['--expansion-fixed-policy',fixed_policy]
                return subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            for name,steps,resume in [('prefix',2,None),('resumed',4,'prefix'),('whole',4,None)]:
                result=execute(name,steps,resume);self.assertEqual(result.returncode,0,result.stderr)
            for level in (0,1):
                np.testing.assert_array_equal(np.load(root/'resumed'/f'velpro_L{level}'/'positions.npy'),
                                              np.load(root/'whole'/f'velpro_L{level}'/'positions.npy'))
            result=execute('changed_fixed_policy',4,'prefix','coarse-indices')
            self.assertNotEqual(result.returncode,0);self.assertIn('different expansion_fixed_policy',result.stderr)
            history.write_text(history.read_text()+'\n')
            result=execute('changed_history',4,'prefix')
            self.assertNotEqual(result.returncode,0);self.assertIn('expansion history mismatch',result.stderr)

    def test_area_filter_allows_thin_expansion_but_stops_interior_collapse(self):
        x=np.array([[0.,0,0],[1.,0,0],[.5,1e-6,0]])
        faces=np.array([[0,1,2]]);d=np.array([[0.,0,0],[1.,0,0],[.5,1e-6,0]])
        self.assertEqual(rank_step(x,d,faces),1.)
        d=np.zeros_like(x);d[2,1]=-2e-6
        # Both endpoints have positive area but the segment collapses at 1/2.
        alpha=rank_step(x,d,faces)
        self.assertGreater(alpha,.4);self.assertLess(alpha,.5)

    def test_history_reconstruction_independence_and_orientation(self):
        h,P,record=fixture();self.assertEqual(len(h.patches),14)
        x,rest,faces=h.initial(h.coarse_rest)
        initial=h.dynamic_prolongation(rest,faces,set())
        # Independent native author queries vs Python dynamic queries.
        np.testing.assert_allclose(initial.U.toarray(),P.U.toarray(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(initial.position(h.coarse_rest),h.fine_rest,atol=1e-12)
        expanded=set()
        for color in h.colors:
            h.update_topology(x,rest,faces,color);expanded.update(color)
            dynamic=h.dynamic_prolongation(rest,faces,expanded)
            np.testing.assert_allclose(dynamic.position(rest[np.unique(list(faces.values()))]),h.fine_rest,atol=1e-12)
        self.assertGreater(max(map(len,h.colors)),1)
        for color in h.colors:
            used=set()
            for i in color:
                ids=set(h.patches[i].ids);self.assertFalse(used & ids);used.update(ids)
        with self.assertRaisesRegex(ValueError,'oriented'):
            ExpansionHistory(record,h.coarse_rest,h.coarse_faces,h.fine_rest,h.fine_faces[:,::-1])

    def test_rest_recovery_through_actual_expansions(self):
        h,P,_=fixture();x,r=initialize(h,h.coarse_rest,P.position(h.coarse_rest))
        self.assertTrue(r['accepted']);self.assertEqual(len(r['batches']),len(h.colors))
        np.testing.assert_allclose(x,h.fine_rest,atol=1e-12,rtol=0)

    def test_safe_self_contact_where_direct_prolongation_crosses(self):
        h,P,record=fixture()
        # A second cloth component belongs to the SAME collision mesh; it is
        # close only in the deformed state, and is not an external obstacle.
        plane=np.array([[-2.,-1.,-.2],[2.,-1.,-.2],[2.,2.,-.2],[-2.,2.,-.2]])
        pf=np.array([[0,1,2],[0,2,3]])
        nf,nc=len(h.fine_rest),len(h.coarse_rest)
        record['fine_vertex_count']+=4;record['coarse_ids']+=list(range(nf,nf+4))
        extended=ExpansionHistory(record,np.vstack((h.coarse_rest,plane)),np.vstack((h.coarse_faces,pf+nc)),
                                  np.vstack((h.fine_rest,plane)),np.vstack((h.fine_faces,pf+nf)))
        coarse=extended.coarse_rest.copy();coarse[:nc,2]=0;coarse[nc:,2]=-.002
        naive=np.vstack((P.position(coarse[:nc]),coarse[nc:]))
        self.assertLess(naive[:nf,2].min(),-.002)
        result,report=initialize(extended,coarse,naive)
        self.assertTrue(report['accepted']);self.assertLess(report['max_stretch'],1.1)
        self.assertGreater(report['final_gap_lower_bound'],.0002)
        self.assertGreater(result[:nf,2].min(),-.002+.0002)
        np.testing.assert_array_equal(result[nf:],coarse[nc:])
        self.assertGreater(sum(len(b['relaxations']) for b in report['batches']),0)

    def test_random_fallback_is_seeded_and_failure_is_bounded(self):
        h,_,_=fixture();color=h.colors[0];p=h.patches[color[0]]
        x,r,f=h.initial(h.coarse_rest);h.update_topology(x,r,f,color)
        target=np.tile(x[p.survivor]+[0,0,.1],(2,1))
        a,b=x.copy(),x.copy()
        ra=safe_perturb(a,p.split,target,p.pre_faces,1e-5,np.random.default_rng(8),128)
        rb=safe_perturb(b,p.split,target,p.pre_faces,1e-5,np.random.default_rng(8),128)
        self.assertTrue(ra['random_fallback']);self.assertEqual(ra,rb)
        np.testing.assert_array_equal(a,b)
        with patch('progressive.refinement_contact.ipctk.has_intersections',return_value=True):
            with self.assertRaisesRegex(RefinementFailure,'within 3 attempts'):
                safe_perturb(x,p.split,target,p.pre_faces,1e-5,np.random.default_rng(8),2)

    def test_ratio_ccd_stops_only_limiting_stencils(self):
        x=np.array([[0.,0,0],[2.,0,0],[0.,2.,0],[.2,.2,.1],[.5,.5,.1]])
        faces=np.array([[0,1,2]]);mesh=collision_mesh(x,faces)
        target=x.copy();target[3,2]=-.1;target[4,2]=.05
        result,records=ratio_advance(mesh,x,target,.1,ExpansionOptions())
        self.assertGreater(result[3,2],0)
        np.testing.assert_allclose(result[4],target[4],atol=1e-14)
        self.assertTrue(any(3 in r['stopped_vertices'] for r in records))
        # Below-gap branch retains at least 99.9% of pair's starting distance.
        x[3,2]=.001;target=x.copy();target[3,2]=-.1
        result,records=ratio_advance(mesh,x,target,.1,ExpansionOptions())
        self.assertGreaterEqual(result[3,2],.999*.001-1e-12)
        self.assertTrue(any(r['small_gap_pairs'] for r in records))

    def test_exact_arap_hessian_and_variational_strain_correction(self):
        m=grid(2);s=Shell(m.rest,m.faces)
        x=m.rest.copy();x[4,0]+=.25
        e,g,H=arap_local(s,x[s.faces],1000.,True)
        np.testing.assert_allclose(H,local_hessian(lambda y:arap_local(s,y,1000.)[:2],x[s.faces],1e-6),rtol=2e-5,atol=2e-4)
        self.assertGreater(s.max_stretch(x),1.1)
        fixed=np.ones(len(x),bool);fixed[4]=False
        out,records=strain_relaxation(s,x,x,fixed,None,ExpansionOptions(relaxation_iterations=100))
        self.assertTrue(s.feasible(out));self.assertGreater(len(records),0)
        np.testing.assert_array_equal(out[fixed],x[fixed])
        self.assertTrue(all(a['infeasible_faces']>=b['infeasible_faces'] for a,b in zip(records,records[1:])))

    def test_fixed_inherited_edge_infeasibility_is_certified_before_iterations(self):
        rest=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]])
        shell=Shell(rest,np.array([[0,1,2]]))
        x=rest.copy();x[1,0]=1.2
        with self.assertRaisesRegex(RefinementFailure,'Fixed inherited endpoints') as caught:
            strain_relaxation(shell,x,rest,np.array([True,True,False]),None,ExpansionOptions())
        self.assertEqual(caught.exception.diagnostics['phase'],'strain_correction_preflight')
        self.assertEqual(caught.exception.diagnostics['fixed_edge_vertices'],[[0,1]])
        self.assertAlmostEqual(caught.exception.diagnostics['fixed_edge_stretches'][0],1.2)

    def test_bad_coarse_contact_and_failed_relaxation_never_accepted(self):
        h,P,_=fixture();x=h.coarse_rest.copy();x[:,2]=0
        plane=np.array([[-2.,-1.,0],[2.,-1.,0],[2.,2.,0],[-2.,2.,0]])
        with self.assertRaises(RefinementFailure) as caught:
            initialize(h,x,P.position(x),obstacles=[(plane,np.array([[0,1,2],[0,2,3]]))])
        self.assertFalse(caught.exception.report['accepted'])


if __name__=='__main__':unittest.main()
