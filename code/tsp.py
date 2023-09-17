import random


class TSP:
    def __init__(self, distance_matrix) -> None:
        self.distance_matrix = distance_matrix
        self.n_points = distance_matrix.shape[0]

    def solve(self):
        pass


class GeneticSolution:
    def __init__(self, genes: list[int]) -> None:
        self.genes = genes

    def calc_fitness(self, distance_matrix):
        fitness = 0

        # calculate the distance of the travelled way defined by the solutions genes
        for i in range(1, len(self.genes)):
            d = distance_matrix[self.genes[i - 1]][self.genes[i]]
            fitness += d

        # this doesnt account for the way from the last station back to the beginning ...
        fitness += distance_matrix[self.genes[0]][self.genes[-1]]

        self.fitness = fitness

    def mutate(self, mutation_chance: float):
        # mutation is performed by randomly switching two positions in the genes array
        pass


class TSP_genetic_algo(TSP):
    def __init__(
        self,
        distance_matrix,
        n_individuals: int,
        n_generations: int,
        ratio_of_individuals_to_survive_per_generation: float,
    ) -> None:
        super().__init__(distance_matrix)

        self.n_generations = n_generations
        self.n_individuals = n_individuals

        self.current_generation_i = 0
        self.current_generation_individuals: list[GeneticSolution]
        self.ratio_of_individuals_to_survive_per_generation = (
            ratio_of_individuals_to_survive_per_generation
        )

    def generate_random_individual(self):
        genes = random.shuffle([*range(0, self.n_points, 1)])
        return GeneticSolution(genes)

    def generate_initial_generation(self):
        generation = []
        for _ in range(self.n_generations):
            generation.append(self.generate_random_individual())

        self.current_generation_individuals = generation

    def breed_new_generation(self):
        # many possibilities
        # in this implementation, we keep x% of the best individuals of the current generation, the rest dies
        # but before "dying", each individual has the chance to produce offspring, the change depening on their fitness

        pass

    def evaluate_generation_fitness(self):
        fitnesses = []
        for ind in self.current_generation_individuals:
            ind.calc_fitness(self.distance_matrix)
            fitnesses.append(ind.fitness)

        self.current_fitnesses = fitnesses

    def crossover_two_individuals(self, ind_1: GeneticSolution, ind_2: GeneticSolution):
        resulting_gene: list[int] = []

        for i in range(self.n_points):
            genes_to_pick = []
            if ind_1.genes[i] not in resulting_gene:
                genes_to_pick.append(ind_1.genes[i])

            if ind_2.genes[i] not in resulting_gene:
                genes_to_pick.append(ind_2.genes[i])

            resulting_gene.append(random.randint(0, len(genes_to_pick)))

        resulting_individual = GeneticSolution(resulting_gene)
        return resulting_individual
