"""Small integration checks for reproducible, cached homework experiments."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fake_followers import experiments as ex


class ExperimentTests(unittest.TestCase):
    def test_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, 'experiment.pkl'))
            args = dict(path=path, recommenders=[('general', 0)], budget_profiles=[{2: 3}],
                        no_seeds=2, num_users=20, num_ccs=4, num_steps=5,
                        record_at_timesteps=(2, 5))
            original, seekers = ex.run_experiment(**args)
            self.assertEqual(len(original), 32)
            self.assertEqual(len(seekers), 8)
            # Pairing is exact under UR because purchases cannot change recommendations.
            frames = [g.sort_values(['random_seed', 'timestep', 'creator'])
                      for _, g in original.groupby('buying_creators', sort=False)]
            self.assertEqual(frames[0].real_followers.tolist(), frames[1].real_followers.tolist())

            # the same experiment is loaded, not simulated again
            with patch.object(ex.simulation, 'simulate', side_effect=AssertionError('cache miss')):
                cached, _ = ex.run_experiment(**args)
            pd.testing.assert_frame_equal(original, cached)

            # an unreadable cache is simulated again, with the same seeds
            Path(path).write_bytes(b'not a pickle')
            repaired, _ = ex.run_experiment(**args)
            pd.testing.assert_frame_equal(original, repaired)

            # changed arguments are simulated again
            changed, _ = ex.run_experiment(**{**args, 'no_seeds': 3})
            self.assertEqual(changed.random_seed.nunique(), 3)

    def test_income_ties_conserve_prizes(self):
        income = ex.creator_income([100, 0, 0, 0], 2, 100)
        self.assertAlmostEqual(sum(income), 150)
        self.assertAlmostEqual(income[1], 50 / 3)


if __name__ == '__main__':
    unittest.main()
