import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

@dataclass
class Device:
    id: int
    x: float
    y: float

@dataclass
class Individual:
    # A candidate solution: list of 5 AP coordinates [(x1, y1), ..., (x5, y5)]
    aps: List[Tuple[float, float]]
    # Cached evaluation metrics
    total_cost: float = float('inf')
    power_cost: float = float('inf')
    overlap_cost: float = float('inf')
    capacity_cost: float = float('inf')
    # Device assignments: list of AP indices (length = number of devices)
    device_assignments: List[int] = field(default_factory=list)
    # AP loads: number of devices connected to each AP (length = number of APs)
    ap_loads: List[int] = field(default_factory=list)

def generate_devices(num_devices: int = 100, grid_size: int = 100, seed: int = 42) -> List[Device]:
    """
    Generates devices randomly using a hardcoded seed.
    Places at most 1 device per grid square (center of the square).
    """
    rng = random.Random(seed)
    all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    # Randomly select unique cells
    chosen_cells = rng.sample(all_cells, num_devices)
    
    # Place each device at the center of the chosen cell (x + 0.5, y + 0.5)
    devices = []
    for i, cell in enumerate(chosen_cells):
        devices.append(Device(id=i, x=cell[0] + 0.5, y=cell[1] + 0.5))
    return devices

class GeneticAlgorithm:
    def __init__(
        self,
        devices: List[Device],
        num_aps: int = 5,
        grid_size: int = 100,
        pop_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
        ap_radius: float = 25.0,
        ap_capacity: int = 22,
        power_weight: float = 1.0,
        overlap_weight: float = 100.0,
        capacity_weight: float = 500.0,
        power_exponent: float = 2.0
    ):
        self.devices = devices
        self.num_aps = num_aps
        self.grid_size = grid_size
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.ap_radius = ap_radius
        self.ap_capacity = ap_capacity
        self.power_weight = power_weight
        self.overlap_weight = overlap_weight
        self.capacity_weight = capacity_weight
        self.power_exponent = power_exponent
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_history: List[float] = []
        self.avg_history: List[float] = []
        
        # Internal random generator for GA operations (independent of device generation seed)
        self.rng = random.Random()
        
        self.initialize_population()

    def initialize_population(self):
        """Creates an initial population of random individuals."""
        self.population = []
        for _ in range(self.pop_size):
            aps = []
            for _ in range(self.num_aps):
                x = self.rng.uniform(0, self.grid_size)
                y = self.rng.uniform(0, self.grid_size)
                aps.append((x, y))
            ind = Individual(aps=aps)
            self.evaluate_individual(ind)
            self.population.append(ind)
        self.generation = 0
        self.best_history = []
        self.avg_history = []
        self.sort_population()
        self.record_stats()

    def evaluate_individual(self, ind: Individual):
        """Calculates the fitness and individual costs for a single individual."""
        # 1. Device assignment & Power Consumption Cost
        power_cost = 0.0
        device_assignments = []
        ap_loads = [0] * self.num_aps
        
        for device in self.devices:
            min_dist = float('inf')
            best_ap = 0
            for ap_idx, ap in enumerate(ind.aps):
                dist = math.hypot(device.x - ap[0], device.y - ap[1])
                if dist < min_dist:
                    min_dist = dist
                    best_ap = ap_idx
            
            # Record assignment
            device_assignments.append(best_ap)
            ap_loads[best_ap] += 1
            
            # Power consumption is proportional to distance^exponent
            power_cost += min_dist ** self.power_exponent
            
        # 2. Access Point Overlap Cost
        overlap_cost = 0.0
        # Check all unique pairs of access points
        for i in range(self.num_aps):
            for j in range(i + 1, self.num_aps):
                dist = math.hypot(ind.aps[i][0] - ind.aps[j][0], ind.aps[i][1] - ind.aps[j][1])
                # Overlap occurs if distance is less than 2 * radius
                overlap_distance = max(0.0, 2.0 * self.ap_radius - dist)
                # Penalty is squared to encourage separation
                overlap_cost += overlap_distance ** 2

        # 3. Capacity Cost
        capacity_cost = 0.0
        for load in ap_loads:
            oversubscription = max(0, load - self.ap_capacity)
            # Penalty is squared to heavily penalize single AP overloading
            capacity_cost += oversubscription ** 2

        # Total cost calculation
        total_cost = (
            self.power_weight * power_cost +
            self.overlap_weight * overlap_cost +
            self.capacity_weight * capacity_cost
        )
        
        # Cache results in individual
        ind.total_cost = total_cost
        ind.power_cost = power_cost
        ind.overlap_cost = overlap_cost
        ind.capacity_cost = capacity_cost
        ind.device_assignments = device_assignments
        ind.ap_loads = ap_loads

    def sort_population(self):
        """Sorts the population in place by total cost (ascending - lower cost is better)."""
        self.population.sort(key=lambda x: x.total_cost)

    def record_stats(self):
        """Records statistics of the current generation."""
        best_cost = self.population[0].total_cost
        avg_cost = sum(ind.total_cost for ind in self.population) / len(self.population)
        self.best_history.append(best_cost)
        self.avg_history.append(avg_cost)

    def select_parent(self, tournament_size: int = 3) -> Individual:
        """Selects an individual using tournament selection."""
        candidates = self.rng.sample(self.population, tournament_size)
        return min(candidates, key=lambda x: x.total_cost)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Performs crossover between two parents to produce two children."""
        if self.rng.random() > self.crossover_rate:
            return Individual(aps=list(parent1.aps)), Individual(aps=list(parent2.aps))
            
        child1_aps = []
        child2_aps = []
        
        # Uniform crossover of AP locations
        for i in range(self.num_aps):
            if self.rng.random() < 0.5:
                child1_aps.append(parent1.aps[i])
                child2_aps.append(parent2.aps[i])
            else:
                child1_aps.append(parent2.aps[i])
                child2_aps.append(parent1.aps[i])
                
        return Individual(aps=child1_aps), Individual(aps=child2_aps)

    def mutate(self, individual: Individual):
        """Mutates an individual's AP coordinates in place."""
        new_aps = []
        for ap in individual.aps:
            if self.rng.random() < self.mutation_rate:
                # With small probability, completely reset the AP (global exploration)
                if self.rng.random() < 0.1:
                    x = self.rng.uniform(0, self.grid_size)
                    y = self.rng.uniform(0, self.grid_size)
                else:
                    # Otherwise, perturb slightly with Gaussian noise (local exploitation)
                    # Mutation step size standard deviation is 5% of grid size
                    std_dev = self.grid_size * 0.05
                    x = self.rng.gauss(ap[0], std_dev)
                    y = self.rng.gauss(ap[1], std_dev)
                    # Clamp coordinates to grid boundary
                    x = max(0.0, min(self.grid_size, x))
                    y = max(0.0, min(self.grid_size, y))
                new_aps.append((x, y))
            else:
                new_aps.append(ap)
        individual.aps = new_aps

    def step(self):
        """Executes one generation step of the Genetic Algorithm."""
        new_pop: List[Individual] = []
        
        # 1. Elitism: Keep the best individuals unchanged
        for i in range(self.elitism_count):
            if i < len(self.population):
                # Copy the individual structure, evaluate to refresh weights if they changed
                elite = Individual(aps=list(self.population[i].aps))
                self.evaluate_individual(elite)
                new_pop.append(elite)
                
        # 2. Reproduction, Crossover, and Mutation
        while len(new_pop) < self.pop_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            self.evaluate_individual(child1)
            self.evaluate_individual(child2)
            
            new_pop.append(child1)
            if len(new_pop) < self.pop_size:
                new_pop.append(child2)
                
        self.population = new_pop
        self.sort_population()
        self.generation += 1
        self.record_stats()

    def get_best_individual(self) -> Individual:
        """Returns the best individual in the current population."""
        return self.population[0]

    def update_parameters(self, config: Dict[str, Any]):
        """Updates GA parameters and re-evaluates the current population with new parameters."""
        if 'pop_size' in config:
            self.pop_size = int(config['pop_size'])
        if 'mutation_rate' in config:
            self.mutation_rate = float(config['mutation_rate'])
        if 'crossover_rate' in config:
            self.crossover_rate = float(config['crossover_rate'])
        if 'elitism_count' in config:
            self.elitism_count = int(config['elitism_count'])
        if 'ap_radius' in config:
            self.ap_radius = float(config['ap_radius'])
        if 'ap_capacity' in config:
            self.ap_capacity = int(config['ap_capacity'])
        if 'power_weight' in config:
            self.power_weight = float(config['power_weight'])
        if 'overlap_weight' in config:
            self.overlap_weight = float(config['overlap_weight'])
        if 'capacity_weight' in config:
            self.capacity_weight = float(config['capacity_weight'])
        if 'power_exponent' in config:
            self.power_exponent = float(config['power_exponent'])
            
        # Re-evaluate all individuals with the new parameters
        for ind in self.population:
            self.evaluate_individual(ind)
        self.sort_population()
        # Correct the last recorded best/average history to match the updated parameters
        if self.best_history:
            self.best_history[-1] = self.population[0].total_cost
            self.avg_history[-1] = sum(ind.total_cost for ind in self.population) / len(self.population)

    def set_devices(self, devices: List[Device], sort: bool = False):
        """Updates the device list and re-evaluates the population without reshuffling unless sort=True."""
        self.devices = devices
        for ind in self.population:
            self.evaluate_individual(ind)
        if sort:
            self.sort_population()
        if self.best_history:
            self.best_history[-1] = self.population[0].total_cost
            self.avg_history[-1] = sum(ind.total_cost for ind in self.population) / len(self.population)

