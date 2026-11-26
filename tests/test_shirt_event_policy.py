import copy
import unittest
from shirt_demo.solver_policy import converged_step, validate_parent, configure_event_solver


class EventPolicyTest(unittest.TestCase):
    def test_rejects_unconverged_and_conflicting_native_status(self):
        ok=dict(completed=True,converged=True,termination_reason='newton_converged')
        self.assertTrue(converged_step(ok))
        for k in ('preview_non_descent','preview_small_step','hit_newton_limit','hit_line_search_limit'):
            self.assertFalse(converged_step(dict(ok,**{k:True})))
        for change in (dict(completed=False),dict(converged=False),dict(termination_reason='preview_non_descent')):
            self.assertFalse(converged_step(dict(ok,**change)))

    def test_old_or_mixed_parent_rejected(self):
        row={'native':dict(completed=True,converged=True,termination_reason='newton_converged')}
        meta=dict(completed=True,event_fallback_enabled=True,requires_convergence=True,
                  event_runtime_sha256='same',completed_steps=1)
        validate_parent(meta,[row],'same')
        for change in (dict(event_fallback_enabled=False),dict(event_runtime_sha256='other'),dict(completed_steps=2)):
            with self.assertRaises(ValueError):validate_parent(dict(meta,**change),[row],'same')
        with self.assertRaises(ValueError):validate_parent(meta,[{'native':dict(row['native'],preview_small_step=True)}],'same')

    def test_tolerances_and_semi_implicit(self):
        cfg=dict(linear_system={},newton={'semi_implicit':{'enable':1}})
        configure_event_solver(cfg,None,None,1500)
        self.assertEqual(cfg['linear_system']['tol_rate'],1e-10)
        self.assertEqual(cfg['newton']['semi_implicit']['enable'],0)
        for linear,newton,budget in ((.001,.001,1500),(1e-10,.05,1500),(float('nan'),.001,1500),(1e-10,.001,0)):
            with self.assertRaises(ValueError):configure_event_solver(copy.deepcopy(cfg),linear,newton,budget)


if __name__=='__main__':unittest.main()
