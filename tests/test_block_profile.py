from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import torch
from wan22.infer.block_profile import FirstBlockProfile


class BlockProfileTests(unittest.TestCase):
    def pipe(self):
        block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())
        return SimpleNamespace(transformer=SimpleNamespace(blocks=[block]), transformer_2=None), block

    def test_profiles_real_block_and_stops_before_next_operation(self):
        pipe, block = self.pipe()
        with tempfile.TemporaryDirectory() as directory:
            with FirstBlockProfile(pipe, Path(directory), cuda=False) as diagnostic:
                block(torch.ones(1, 3, 8))
                self.fail('must stop after the first block')
            result = json.loads((Path(directory) / 'block-profile.json').read_text())
            self.assertEqual(result['inputs']['args'][0]['shape'], [1, 3, 8])
            self.assertTrue(result['timing_valid'])
            self.assertTrue(any(row['name'] == 'wan_block/0' for row in result['operators']))
            self.assertIsNotNone(diagnostic.result)
            self.assertTrue((Path(directory) / 'block-trace.json').is_file())
        for module in block.modules():
            self.assertNotIn('forward', module.__dict__)
        self.assertEqual(block(torch.ones(1, 3, 8)).shape, (1, 3, 8))

    def test_failure_propagates_and_restores_instance_forward(self):
        pipe, block = self.pipe()
        def broken(*args):
            raise ValueError('real failure')
        block[0].forward = broken
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'real failure'):
                with FirstBlockProfile(pipe, Path(directory), cuda=False) as diagnostic:
                    block(torch.ones(1, 3, 8))
            self.assertIsNone(diagnostic.result)
            self.assertIs(block[0].forward, broken)
            self.assertNotIn('forward', block.__dict__)
            self.assertFalse((Path(directory) / 'block-profile.json').exists())

    def test_unused_block_is_not_success(self):
        pipe, block = self.pipe()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'without executing'):
                with FirstBlockProfile(pipe, Path(directory), cuda=False):
                    pass
        self.assertNotIn('forward', block.__dict__)

    def test_setup_failure_restores_already_patched_expert(self):
        pipe, block = self.pipe()
        pipe.transformer_2 = SimpleNamespace(blocks=[])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(IndexError):
                with FirstBlockProfile(pipe, Path(directory), cuda=False):
                    pass
        self.assertTrue(all('forward' not in module.__dict__ for module in block.modules()))
