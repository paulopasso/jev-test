import unittest
from unittest.mock import patch
import numpy as np
import torch
from jev_research import evaluate, config
from jev_research.train_annot import EpisodeData

class Episodes:
    def __init__(self):
        self.acts = {'demo': np.arange(1, 31, dtype=np.float32).reshape(-1, 1)}
        f = np.concatenate([np.zeros((1, 1), np.float32), self.acts['demo'].cumsum(0)])
        self.eps = {'demo': {'feat': f}}
    def norm_feat(self, f): return f
    def norm_act(self, a): return (a - 5) / 2

class Dynamics(torch.nn.Module):
    def forward(self, ctx, a): return ctx[:, -1] + a * 2 + 5

class IntegrityTests(unittest.TestCase):
    def test_exact_dynamics_and_alignment(self):
        with patch.object(evaluate, 'DEV', 'cpu'):
            macro, _ = evaluate.rollout_error(Dynamics(), Episodes(), ['demo'], [1,5,10,20], 4)
        self.assertEqual(macro['mean_1_10'], 0)
        self.assertEqual(macro[20], 0)
    def test_native_zero_control(self):
        eps = Episodes()
        with patch.object(evaluate, 'DEV', 'cpu'):
            value = evaluate.control_rollouts(Dynamics(), eps, ['demo'], 4, 'zero')
        f = eps.eps['demo']['feat']
        expected = np.mean((f[4:14]-f[3])**2)
        self.assertAlmostEqual(value, float(expected), places=4)
    def test_train_only_statistics(self):
        eps = EpisodeData.__new__(EpisodeData)
        eps.eps = {'train': {'feat': np.ones((2,3),np.float32)},
                   'dev': {'feat': np.full((2,3), 100, np.float32)}}
        with patch('jev_research.data.load_split', return_value={'train':['train']}):
            np.testing.assert_array_equal(eps.feature_stats(['train'])['mean'], np.ones(3))
            with self.assertRaises(ValueError): eps.feature_stats(['train','dev'])
    def test_short_episode_fails_instead_of_changing_endpoint(self):
        with self.assertRaises(ValueError):
            evaluate.rollout_error(Dynamics(), Episodes(), ['demo'], [40], 4)
    def test_candidate_zero_required(self):
        with self.assertRaises(ValueError): evaluate._candidate_bank(np.zeros(7), np.random.default_rng(0), 5)
    def test_nonfinite_ci_rejected(self):
        with self.assertRaises(ValueError): evaluate.bootstrap_ci([0,np.nan])
    def test_stale_receipt_fails_closed(self):
        import tempfile, json, os
        with tempfile.TemporaryDirectory() as d, patch.object(config, 'CACHE_DIR', d), patch.object(config, 'integrity_fingerprint', return_value='current'):
            with open(os.path.join(d, f'{config.TASK}_e0_receipt.json'), 'w') as f:
                json.dump({'pass':True,'fingerprint':'old'},f)
            with self.assertRaises(RuntimeError): config.require_integrity()

if __name__ == '__main__': unittest.main()
