import random
import math
import os
import concurrent.futures
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

def _eval_single_individual_worker(args: Tuple) -> Tuple[float, float, float, float, List[int], List[int]]:
    aps, devices, num_aps, ap_radius, ap_capacity, power_weight, overlap_weight, capacity_weight, power_exponent = args
    
    # 1. Device assignment & Power Consumption Cost
    power_cost = 0.0
    device_assignments = []
    ap_loads = [0] * num_aps
    
    for device in devices:
        min_dist = float('inf')
        best_ap = 0
        for ap_idx, ap in enumerate(aps):
            dist = math.hypot(device.x - ap[0], device.y - ap[1])
            if dist < min_dist:
                min_dist = dist
                best_ap = ap_idx
        
        device_assignments.append(best_ap)
        ap_loads[best_ap] += 1
        power_cost += min_dist ** power_exponent
        
    # 2. Access Point Overlap Cost
    overlap_cost = 0.0
    for i in range(num_aps):
        for j in range(i + 1, num_aps):
            dist = math.hypot(aps[i][0] - aps[j][0], aps[i][1] - aps[j][1])
            overlap_distance = max(0.0, 2.0 * ap_radius - dist)
            overlap_cost += overlap_distance ** 2

    # 3. Capacity Cost
    capacity_cost = 0.0
    for load in ap_loads:
        oversubscription = max(0, load - ap_capacity)
        capacity_cost += oversubscription ** 2

    total_cost = (
        power_weight * power_cost +
        overlap_weight * overlap_cost +
        capacity_weight * capacity_cost
    )
    
    return (total_cost, power_cost, overlap_cost, capacity_cost, device_assignments, ap_loads)

def generate_devices(num_devices: int = 100, grid_size: int = 100, seed: int = 42) -> List[Device]:
    """
    Generates devices randomly using a hardcoded seed.
    Places at most 1 device per grid square (on grid line intersections).
    """
    rng = random.Random(seed)
    all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    # Randomly select unique cells
    chosen_cells = rng.sample(all_cells, num_devices)
    
    # Place each device on integer grid line coordinates (x, y)
    devices = []
    for i, cell in enumerate(chosen_cells):
        devices.append(Device(id=i, x=float(cell[0]), y=float(cell[1])))
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
        power_weight: float = 40.0,
        overlap_weight: float = 30.0,
        capacity_weight: float = 500.0,
        power_exponent: float = 2.0,
        max_generations: int = 1000
    ):
        self.devices = devices
        self.num_aps = num_aps
        self.grid_size = grid_size
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate  # baseline for adaptive mutation
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.ap_radius = ap_radius
        self.ap_capacity = ap_capacity
        self.power_weight = power_weight
        self.overlap_weight = overlap_weight
        self.capacity_weight = capacity_weight
        self.power_exponent = power_exponent
        self.max_generations = max_generations
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_history: List[float] = []
        self.avg_history: List[float] = []
        
        # --- Anti-local-minima parameters ---
        # Stagnation detection
        self.stagnation_counter: int = 0
        self.stagnation_threshold: int = 20  # generations without improvement to trigger adaptation
        self.stagnation_improvement_pct: float = 0.001  # 0.1% improvement required to reset stagnation
        
        # Adaptive mutation
        self.adaptive_mutation_boost: float = 3.0  # multiplier applied to mutation rate during stagnation
        self.mutation_std_scale: float = 0.05  # current mutation step size (fraction of grid_size)
        self.base_mutation_std_scale: float = 0.05  # baseline step size
        self.stagnation_std_scale: float = 0.20  # larger step size used during stagnation
        
        # Random immigrant injection
        self.immigrant_fraction: float = 0.15  # fraction of population replaced with random individuals on stagnation
        
        # Blend crossover alpha (BLX-α)
        self.blx_alpha: float = 0.3  # exploration range beyond parents
        
        # Internal random generator for GA operations
        self.rng = random.Random()
        
        # Multiprocessing CPU Pool Executor
        self.num_workers = max(1, os.cpu_count() or 4)
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        
        self.initialize_population()

    @property
    def is_finished(self) -> bool:
        """Returns True if the genetic algorithm has reached max_generations."""
        return self.generation >= self.max_generations

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
            self.population.append(ind)
        self.evaluate_population(self.population)
        self.generation = 0
        self.best_history = []
        self.avg_history = []
        self.sort_population()
        self.record_stats()

    def evaluate_population(self, inds: List[Individual]):
        """Evaluates multiple individuals in parallel using all available CPU cores."""
        if not inds:
            return
        if self.num_workers > 1 and len(inds) >= 4:
            tasks = [
                (
                    ind.aps,
                    self.devices,
                    self.num_aps,
                    self.ap_radius,
                    self.ap_capacity,
                    self.power_weight,
                    self.overlap_weight,
                    self.capacity_weight,
                    self.power_exponent
                )
                for ind in inds
            ]
            chunk = max(1, len(inds) // (self.num_workers * 2))
            try:
                results = list(self.executor.map(_eval_single_individual_worker, tasks, chunksize=chunk))
                for ind, res in zip(inds, results):
                    (
                        ind.total_cost,
                        ind.power_cost,
                        ind.overlap_cost,
                        ind.capacity_cost,
                        ind.device_assignments,
                        ind.ap_loads
                    ) = res
            except Exception:
                for ind in inds:
                    self.evaluate_individual(ind)
        else:
            for ind in inds:
                self.evaluate_individual(ind)

    def evaluate_individual(self, ind: Individual):
        """Calculates the fitness and individual costs for a single individual."""
        res = _eval_single_individual_worker((
            ind.aps,
            self.devices,
            self.num_aps,
            self.ap_radius,
            self.ap_capacity,
            self.power_weight,
            self.overlap_weight,
            self.capacity_weight,
            self.power_exponent
        ))
        (
            ind.total_cost,
            ind.power_cost,
            ind.overlap_cost,
            ind.capacity_cost,
            ind.device_assignments,
            ind.ap_loads
        ) = res

    def sort_population(self):
        """Sorts the population in place by total cost (ascending - lower cost is better)."""
        self.population.sort(key=lambda x: x.total_cost)

    def record_stats(self):
        """Records statistics of the current generation."""
        best_cost = self.population[0].total_cost
        avg_cost = sum(ind.total_cost for ind in self.population) / len(self.population)
        self.best_history.append(best_cost)
        self.avg_history.append(avg_cost)

    def select_parent(self, tournament_size: int = 5) -> Individual:
        """Selects an individual using tournament selection."""
        k = min(tournament_size, len(self.population))
        candidates = self.rng.sample(self.population, k)
        return min(candidates, key=lambda x: x.total_cost)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Performs blend crossover (BLX-α) between two parents to produce two children.
        
        BLX-α explores the space between parents and slightly beyond (controlled by
        self.blx_alpha), which helps escape local minima by generating offspring
        that are not simple recombinations of parent genes.
        """
        if self.rng.random() > self.crossover_rate:
            return Individual(aps=list(parent1.aps)), Individual(aps=list(parent2.aps))
            
        child1_aps = []
        child2_aps = []
        alpha = self.blx_alpha
        
        for i in range(self.num_aps):
            x1, y1 = parent1.aps[i]
            x2, y2 = parent2.aps[i]
            
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            
            x_lo = min(x1, x2) - alpha * dx
            x_hi = max(x1, x2) + alpha * dx
            y_lo = min(y1, y2) - alpha * dy
            y_hi = max(y1, y2) + alpha * dy
            
            # Child 1: sample within the extended range
            cx1 = self.rng.uniform(x_lo, x_hi)
            cy1 = self.rng.uniform(y_lo, y_hi)
            cx1 = max(0.0, min(self.grid_size, cx1))
            cy1 = max(0.0, min(self.grid_size, cy1))
            child1_aps.append((cx1, cy1))
            
            # Child 2: sample independently within the same range
            cx2 = self.rng.uniform(x_lo, x_hi)
            cy2 = self.rng.uniform(y_lo, y_hi)
            cx2 = max(0.0, min(self.grid_size, cx2))
            cy2 = max(0.0, min(self.grid_size, cy2))
            child2_aps.append((cx2, cy2))
                
        return Individual(aps=child1_aps), Individual(aps=child2_aps)

    def mutate(self, individual: Individual):
        """Mutates an individual's AP coordinates in place.
        
        Uses adaptive mutation: the effective mutation rate and step size
        are boosted during stagnation periods to increase exploration.
        """
        effective_rate = self.mutation_rate
        effective_std = self.mutation_std_scale
        
        new_aps = []
        for ap in individual.aps:
            if self.rng.random() < effective_rate:
                if self.rng.random() < 0.15:
                    # Full random reset for this AP (increased from 10% to 15%)
                    x = self.rng.uniform(0, self.grid_size)
                    y = self.rng.uniform(0, self.grid_size)
                else:
                    std_dev = self.grid_size * effective_std
                    x = self.rng.gauss(ap[0], std_dev)
                    y = self.rng.gauss(ap[1], std_dev)
                    x = max(0.0, min(self.grid_size, x))
                    y = max(0.0, min(self.grid_size, y))
                new_aps.append((x, y))
            else:
                new_aps.append(ap)
        individual.aps = new_aps

    def _detect_and_adapt_stagnation(self):
        """Detects stagnation and adapts mutation parameters to escape local minima.
        
        Stagnation is detected when the best fitness hasn't improved by at least
        stagnation_improvement_pct for stagnation_threshold consecutive generations.
        When stagnated, mutation rate and step size are boosted.
        """
        if len(self.best_history) < 2:
            return
        
        prev_best = self.best_history[-2]
        curr_best = self.best_history[-1]
        
        # Check if meaningful improvement occurred
        if prev_best > 0 and (prev_best - curr_best) / prev_best > self.stagnation_improvement_pct:
            # Progress is being made — reset stagnation and restore baseline parameters
            self.stagnation_counter = 0
            self.mutation_rate = self.base_mutation_rate
            self.mutation_std_scale = self.base_mutation_std_scale
        else:
            self.stagnation_counter += 1
        
        if self.stagnation_counter >= self.stagnation_threshold:
            # Boost mutation to increase exploration
            self.mutation_rate = min(1.0, self.base_mutation_rate * self.adaptive_mutation_boost)
            self.mutation_std_scale = self.stagnation_std_scale
    
    def _inject_random_immigrants(self, population: List[Individual]) -> List[Individual]:
        """Replaces the worst fraction of the population with fresh random individuals.
        
        This re-injects genetic diversity when the population has converged too tightly,
        giving the GA new material to combine with the existing elite solutions.
        """
        num_immigrants = max(1, int(len(population) * self.immigrant_fraction))
        
        immigrants = []
        for _ in range(num_immigrants):
            aps = []
            for _ in range(self.num_aps):
                x = self.rng.uniform(0, self.grid_size)
                y = self.rng.uniform(0, self.grid_size)
                aps.append((x, y))
            immigrants.append(Individual(aps=aps))
        
        self.evaluate_population(immigrants)
        
        # Replace the worst individuals (end of sorted list) with immigrants
        population[-num_immigrants:] = immigrants
        return population

    def step(self):
        """Executes one generation step of the Genetic Algorithm in parallel.
        
        Includes adaptive mutation, stagnation detection, and random immigrant
        injection to avoid getting trapped in local minima.
        """
        if self.is_finished:
            return
        
        # Adapt mutation parameters based on stagnation detection
        self._detect_and_adapt_stagnation()
            
        new_pop: List[Individual] = []
        
        # 1. Elitism: Keep the best individuals unchanged
        for i in range(self.elitism_count):
            if i < len(self.population):
                elite = Individual(aps=list(self.population[i].aps))
                new_pop.append(elite)
                
        # 2. Reproduction, Crossover, and Mutation
        while len(new_pop) < self.pop_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            new_pop.append(child1)
            if len(new_pop) < self.pop_size:
                new_pop.append(child2)
                
        # Evaluate new population in parallel across all CPU cores
        self.evaluate_population(new_pop)
        self.population = new_pop
        self.sort_population()
        
        # 3. Inject random immigrants when stagnated to re-diversify
        if self.stagnation_counter >= self.stagnation_threshold:
            self.population = self._inject_random_immigrants(self.population)
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
            self.base_mutation_rate = float(config['mutation_rate'])
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
        if 'max_generations' in config:
            self.max_generations = int(config['max_generations'])
        if 'max_iterations' in config:
            self.max_generations = int(config['max_iterations'])
            
        self.evaluate_population(self.population)
        self.sort_population()
        if self.best_history:
            self.best_history[-1] = self.population[0].total_cost
            self.avg_history[-1] = sum(ind.total_cost for ind in self.population) / len(self.population)

    def set_devices(self, devices: List[Device], sort: bool = False):
        """Updates the device list and re-evaluates the population without reshuffling unless sort=True."""
        self.devices = devices
        self.evaluate_population(self.population)
        if sort:
            self.sort_population()
        if self.best_history:
            self.best_history[-1] = self.population[0].total_cost
            self.avg_history[-1] = sum(ind.total_cost for ind in self.population) / len(self.population)

