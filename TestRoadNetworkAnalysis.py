import unittest
from RoadNetwork import RoadNetwork

class TestRoadNetworkAnalysis(unittest.TestCase):
    def setUp(self):
        self.network = RoadNetwork()
        self.network.add_node('H1', 'critical')
        self.network.add_node('N1', 'neighborhood')
        self.network.add_edge('H1', 'N1', 3, 3000)

    def test_cost_calculation(self):
        costs = self.network.calculate_total_cost()
        self.assertEqual(costs['construction_cost'], 3000)
