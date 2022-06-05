# Copyright 2022 Michael Hu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for rating.py"""

from absl.testing import absltest
from absl.testing import parameterized

import math
from alpha_zero import rating


class EloRatingTest(parameterized.TestCase):
    def test_estimate_win_probability(self):
        prob_0 = rating.estimate_win_probability(1613, 1609)
        prob_1 = rating.estimate_win_probability(1613, 1477)
        prob_2 = rating.estimate_win_probability(1613, 1586)
        prob_3 = rating.estimate_win_probability(1613, 1720)

        self.assertAlmostEqual(prob_0, 0.506, 3)
        self.assertAlmostEqual(prob_1, 0.686, 3)
        self.assertAlmostEqual(prob_2, 0.539, 3)
        self.assertAlmostEqual(prob_3, 0.351, 3)

    def test_compute_elo_rating(self):
        ra, rb = rating.compute_elo_rating(0, 1200, 1200)
        self.assertGreater(ra, rb)

        ra, rb = rating.compute_elo_rating(1, 1200, 1200)
        self.assertGreater(rb, ra)


if __name__ == '__main__':
    absltest.main()
