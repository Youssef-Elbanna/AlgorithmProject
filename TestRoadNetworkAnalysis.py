# TestRoadNetworkAnalysis.py
import unittest
from RoadNetwork import RoadNetwork

class TestRoadNetworkAnalysis(unittest.TestCase):
    def setUp(self):
        self.net = RoadNetwork()
        self.net.add_node('1','critical')
        self.net.add_node('5','neighborhood')
        # distance=3, condition=7
        self.net.add_edge('1','5',3,7)

    def test_cost_calculation(self):
        c = self.net.calculate_total_cost()
        self.assertEqual(c['construction_cost'], 3)

    def test_dijkstra(self):
        path, cost = self.net.dijkstra('11','5')
        self.assertEqual(path, ['11','5'])
        self.assertAlmostEqual(cost, 3/7)

    def test_a_star(self):
        path, cost = self.net.a_star('5','1')
        self.assertEqual(path, ['5','1'])
        self.assertAlmostEqual(cost, 3/7)

if __name__ == '__main__':
    unittest.main()
