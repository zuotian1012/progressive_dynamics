import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from progressive.bunny_suite import run


class BunnySuiteTests(unittest.TestCase):
    def test_failed_method_preserves_result_and_controls_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.json';output=root/'suite'
            config.write_text(json.dumps({'parameters':{'scene':'bunny','methods':['semi','direct']}}))
            launched=[]
            def execute(command,log):
                if 'progressive.run' in command:
                    method=command[command.index('--methods')+1];launched.append(method)
                    destination=Path(command[command.index('--output')+1]);destination.mkdir()
                    (destination/'experiment.json').write_text(json.dumps({'completed':method=='direct'}))
                    return 1 if method=='semi' else 0
                return 0
            with patch('progressive.bunny_suite.fingerprint',return_value={'source':'fixed'}),patch('progressive.bunny_suite.execute',side_effect=execute):
                state=run(config,output)
                self.assertEqual(launched,['semi','direct'])
                self.assertEqual([j['status'] for j in state['jobs']],['failed','completed'])
                self.assertTrue(state['finished']);self.assertFalse(state['completed'])
                self.assertNotIn('report_exit_code',state)
                with self.assertRaisesRegex(ValueError,'overwrite'):run(config,output)


if __name__=='__main__':unittest.main()
