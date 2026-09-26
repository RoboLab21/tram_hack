import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

"""
Юнит-тесты для модуля одометрии.
"""

import unittest
from odometry_node import DeadReckoningNode


class TestDeadReckoningNode(unittest.TestCase):
    def test_basic_integration(self):
        # Входные скорости в км/ч: 36 км/ч = 10 м/с
        node = DeadReckoningNode(input_in_kmh=True, integration_method="rectangular")
        
        # t=0, v=36 км/ч
        node.step(0.0, 36.0, 36.0)
        self.assertEqual(node.R, 0.0)
        
        # t=1.0, v=36 км/ч -> R = 10 * 1 = 10 м
        node.step(1.0, 36.0, 36.0)
        self.assertAlmostEqual(node.R, 10.0, places=4)
        
        # t=2.0, v=72 км/ч (20 м/с) -> R = 10 + 20 * 1 = 30 м
        node.step(2.0, 72.0, 72.0)
        self.assertAlmostEqual(node.R, 30.0, places=4)

    def test_asymmetric_bogies(self):
        # Передняя 36 км/ч (10 м/с), задняя 18 км/ч (5 м/с) -> средняя 7.5 м/с
        node = DeadReckoningNode(input_in_kmh=True, integration_method="rectangular")
        node.step(0.0, 36.0, 18.0)
        node.step(1.0, 36.0, 18.0)
        self.assertAlmostEqual(node.v_avg, 7.5, places=4)
        self.assertAlmostEqual(node.R, 7.5, places=4)

    def test_time_jump_protection(self):
        node = DeadReckoningNode(input_in_kmh=True)
        node.step(0.0, 36.0, 36.0)
        # Скачок времени назад
        state = node.step(-1.0, 36.0, 36.0)
        self.assertEqual(state.delta_r, 0.0)
        # Скачок времени вперед более 2 сек (разрыв связи)
        state = node.step(10.0, 36.0, 36.0)
        self.assertEqual(state.delta_r, 0.0)


if __name__ == "__main__":
    unittest.main()
