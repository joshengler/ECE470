import unittest
import math
from genetic_algorithm import generate_devices, GeneticAlgorithm, Individual, Device

class TestWirelessOptimizationGA(unittest.TestCase):

    def setUp(self):
        # Generate devices with hardcoded seed 42
        self.devices = generate_devices(num_devices=100, grid_size=100, seed=42)

    def test_device_generation(self):
        """Verify device generation properties."""
        # Check size
        self.assertEqual(len(self.devices), 100)
        
        # Check that devices are within bounds and have unique squares
        grid_squares = set()
        for dev in self.devices:
            self.assertTrue(0 <= dev.x <= 100)
            self.assertTrue(0 <= dev.y <= 100)
            
            # Since cell centers are at integer + 0.5, we get the cell by rounding down
            cell = (int(dev.x), int(dev.y))
            grid_squares.add(cell)
            
        # Verify exactly 100 unique grid cells are occupied
        self.assertEqual(len(grid_squares), 100)

        # Check reproducibility of seed 42
        devices_2 = generate_devices(num_devices=100, grid_size=100, seed=42)
        for d1, d2 in zip(self.devices, devices_2):
            self.assertEqual(d1.id, d2.id)
            self.assertAlmostEqual(d1.x, d2.x)
            self.assertAlmostEqual(d1.y, d2.y)

    def test_ga_initialization(self):
        """Verify initialization of the genetic algorithm."""
        ga = GeneticAlgorithm(
            devices=self.devices,
            num_aps=5,
            grid_size=100,
            pop_size=12,
            mutation_rate=0.1
        )
        # Verify pop size
        self.assertEqual(len(ga.population), 12)
        
        # Verify each individual has 5 APs within bounds
        for ind in ga.population:
            self.assertEqual(len(ind.aps), 5)
            for ap in ind.aps:
                self.assertTrue(0 <= ap[0] <= 100)
                self.assertTrue(0 <= ap[1] <= 100)

            # Verification of initialized costs
            self.assertNotEqual(ind.total_cost, float('inf'))
            self.assertEqual(len(ind.device_assignments), 100)
            self.assertEqual(len(ind.ap_loads), 5)
            self.assertEqual(sum(ind.ap_loads), 100)

        # History should have 1 entry (generation 0 state)
        self.assertEqual(ga.generation, 0)
        self.assertEqual(len(ga.best_history), 1)
        self.assertEqual(len(ga.avg_history), 1)

    def test_cost_calculation(self):
        """Verify the components of the cost/fitness function."""
        # Setup a GA instance with predictable weights
        ga = GeneticAlgorithm(
            devices=self.devices,
            num_aps=5,
            grid_size=100,
            pop_size=10,
            ap_radius=25.0,
            ap_capacity=20,
            power_weight=1.0,
            overlap_weight=10.0,
            capacity_weight=100.0,
            power_exponent=2.0
        )
        
        # Create an individual with all APs stacked at the exact center (50, 50)
        aps = [(50.0, 50.0) for _ in range(5)]
        ind = Individual(aps=aps)
        ga.evaluate_individual(ind)
        
        # 1. Check Capacity Cost
        # Since all APs are in the same spot, devices will be assigned to AP 0 (due to floating point/min distance)
        # or split if distances are identical, but in our code, the first AP with minimum distance wins.
        # So AP 0 gets all 100 devices. APs 1, 2, 3, 4 get 0.
        # Capacity limit = 20. Overcapacity for AP 0 = 100 - 20 = 80.
        # Capacity penalty = 80^2 * weight = 6400 * 100 = 640,000.
        self.assertEqual(ind.ap_loads[0], 100)
        self.assertEqual(sum(ind.ap_loads[1:]), 0)
        expected_capacity_cost = (80 ** 2) # 6400 (unweighted)
        self.assertAlmostEqual(ind.capacity_cost, expected_capacity_cost)

        # 2. Check Overlap Cost
        # Since there are 5 APs, there are 10 unique pairs.
        # Distance between all pairs is 0.0.
        # Radius R = 25.0. Overlap distance = 2R - 0 = 50.0.
        # Penalty per pair = 50^2 = 2500.
        # Total overlap cost = 10 pairs * 2500 = 25000 (unweighted).
        expected_overlap_cost = 10 * (50.0 ** 2)
        self.assertAlmostEqual(ind.overlap_cost, expected_overlap_cost)

        # 3. Check Power Cost
        # Since all APs are at (50, 50), the distance to AP for device i is distance to (50, 50).
        expected_power_cost = sum(math.hypot(d.x - 50.0, d.y - 50.0)**2 for d in self.devices)
        self.assertAlmostEqual(ind.power_cost, expected_power_cost)

        # 4. Check Total Cost
        expected_total_cost = (
            ga.power_weight * expected_power_cost +
            ga.overlap_weight * expected_overlap_cost +
            ga.capacity_weight * expected_capacity_cost
        )
        self.assertAlmostEqual(ind.total_cost, expected_total_cost)

    def test_crossover(self):
        """Verify crossover combines parents correctly."""
        ga = GeneticAlgorithm(devices=self.devices, pop_size=10)
        
        # Parents with easily distinguishable AP coordinates
        parent1 = Individual(aps=[(10.0, 10.0)] * 5)
        parent2 = Individual(aps=[(90.0, 90.0)] * 5)
        
        # Test crossover under force rate = 1.0
        ga.crossover_rate = 1.0
        child1, child2 = ga.crossover(parent1, parent2)
        
        # The coordinates of children APs must all be either (10.0, 10.0) or (90.0, 90.0)
        for ap in child1.aps:
            self.assertTrue(ap == (10.0, 10.0) or ap == (90.0, 90.0))
        for ap in child2.aps:
            self.assertTrue(ap == (10.0, 10.0) or ap == (90.0, 90.0))
            
        # Verify crossover rate = 0.0 returns copies of parents
        ga.crossover_rate = 0.0
        c1, c2 = ga.crossover(parent1, parent2)
        self.assertEqual(c1.aps, parent1.aps)
        self.assertEqual(c2.aps, parent2.aps)

    def test_mutation(self):
        """Verify mutation logic perturbs coordinates but respects boundaries."""
        ga = GeneticAlgorithm(devices=self.devices, pop_size=10, mutation_rate=1.0)
        # Individual placed at the center (50, 50)
        ind = Individual(aps=[(50.0, 50.0)] * 5)
        
        # Mutate
        ga.mutate(ind)
        
        # Verify that coordinates are still within the grid bounds [0, 100]
        # and that they have been perturbed from (50, 50)
        for ap in ind.aps:
            self.assertTrue(0.0 <= ap[0] <= 100.0)
            self.assertTrue(0.0 <= ap[1] <= 100.0)
            
            # Since mutation rate is 1.0, and perturbation is random gauss,
            # they shouldn't remain exactly (50.0, 50.0)
            self.assertNotEqual(ap, (50.0, 50.0))

    def test_step_execution(self):
        """Verify that step increments generation and improves or maintains best fitness (elitism)."""
        ga = GeneticAlgorithm(
            devices=self.devices,
            pop_size=20,
            elitism_count=2
        )
        
        initial_best_cost = ga.get_best_individual().total_cost
        
        # Take 5 steps
        for i in range(5):
            ga.step()
            self.assertEqual(ga.generation, i + 1)
            self.assertEqual(len(ga.best_history), i + 2)
            self.assertEqual(len(ga.avg_history), i + 2)
            
        final_best_cost = ga.get_best_individual().total_cost
        
        # Elitism guarantees that the best cost cannot worsen (meaning final_best_cost <= initial_best_cost)
        self.assertLessEqual(final_best_cost, initial_best_cost)

    def test_termination_after_max_generations(self):
        """Verify GA terminates after reaching max_generations (1000 by default)."""
        ga = GeneticAlgorithm(
            devices=self.devices,
            pop_size=10,
            max_generations=5
        )
        self.assertEqual(ga.max_generations, 5)
        self.assertFalse(ga.is_finished)
        
        for _ in range(5):
            ga.step()
            
        self.assertEqual(ga.generation, 5)
        self.assertTrue(ga.is_finished)
        
        # Subsequent step should not increment generation beyond max_generations
        ga.step()
        self.assertEqual(ga.generation, 5)
        self.assertTrue(ga.is_finished)

        # Test default max_generations is 1000
        ga_default = GeneticAlgorithm(devices=self.devices, pop_size=10)
        self.assertEqual(ga_default.max_generations, 1000)

if __name__ == '__main__':
    unittest.main()
