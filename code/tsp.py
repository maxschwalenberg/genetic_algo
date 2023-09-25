import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os

import random

from code.utils import get_basemap_params_from_gdf, coordinates_path_from_indices


class TSP:
    def __init__(self, distance_matrix, points, gdf) -> None:
        self.distance_matrix = distance_matrix
        self.n_points = distance_matrix.shape[0]

        self.points = points
        self.gdf = gdf

    def visualize_map(self, output_path: str, solution: list[int]):
        lower_left, upper_right, mid = get_basemap_params_from_gdf(self.gdf, 2)

        fig = plt.figure(figsize=(8, 8))
        m = Basemap(
            llcrnrlon=lower_left[0],
            llcrnrlat=lower_left[1],
            urcrnrlon=upper_right[0],
            urcrnrlat=upper_right[1],
        )
        m.drawcoastlines(linewidth=1.0, linestyle="solid", color="black")
        m.drawcountries(linewidth=1.0, linestyle="solid", color="k")
        m.fillcontinents()

        lons = self.gdf.lon.tolist()
        lats = self.gdf.lat.tolist()

        coordinates = np.column_stack((lons, lats))
        path_lons, path_lats = coordinates_path_from_indices(solution, coordinates)
        x, y = m(lons, lats)
        x_path, y_path = m(path_lons, path_lats)

        m.scatter(x, y)
        m.plot(x_path, y_path)

        plt.title("TSP", fontsize=20)
        plt.savefig(output_path, bbox_inches="tight")

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
        for gene_i, gene in enumerate(self.genes):
            random_p = random.uniform(0, 1)
            if random_p <= mutation_chance:
                # mutate
                switch_with = random.randint(0, len(self.genes) - 1)
                other_gene = self.genes[switch_with]
                self.genes[switch_with] = gene
                self.genes[gene_i] = other_gene

            else:
                continue


class TSPGeneticAlgo(TSP):
    def __init__(
        self,
        distance_matrix,
        points,
        gdf,
        n_individuals: int,
        n_generations: int,
        ratio_of_individuals_to_survive_per_generation: float,
        mutation_rate: float,
        visualization_output_dir_path: str,
    ) -> None:
        super().__init__(distance_matrix, points, gdf)

        self.n_generations = n_generations
        self.n_individuals = n_individuals

        self.current_generation_i = 0
        self.current_generation_individuals: list[GeneticSolution]
        self.ratio_of_individuals_to_survive_per_generation = (
            ratio_of_individuals_to_survive_per_generation
        )
        self.mutation_rate = mutation_rate
        self.visualization_output_dir_path = visualization_output_dir_path

    def generate_random_individual(self):
        genes = [*range(0, self.n_points, 1)]
        random.shuffle(genes)
        return GeneticSolution(genes)

    def generate_initial_generation(self):
        generation = []
        for _ in range(self.n_generations):
            generation.append(self.generate_random_individual())

        self.current_generation_individuals = generation

    def breed_new_generation(self):
        processed_fitnesses = []
        best, worst = (min(self.current_fitnesses), max(self.current_fitnesses))
        for i in range(len(self.current_fitnesses)):
            processed_fitnesses.append(self.current_fitnesses[i] - best)

        new_generation = []

        # many possibilities
        # in this implementation, we keep x% of the best individuals of the current generation, the rest dies
        # but before "dying", each individual has the chance to produce offspring, the change depening on their fitness
        n_to_survive = int(
            self.ratio_of_individuals_to_survive_per_generation * self.n_individuals
        )
        n_to_get_by_crossover = self.n_individuals - n_to_survive

        # take the n best individuals
        current_fitnesses_sorted_inds = np.array(processed_fitnesses).argsort()
        current_generation_individuals_sorted = np.array(
            self.current_generation_individuals
        )[current_fitnesses_sorted_inds]
        current_fitnesses_sorted = np.array(processed_fitnesses)[
            current_fitnesses_sorted_inds
        ]

        max_fitness = max(current_fitnesses_sorted)
        probs_from_fitness = [
            (max_fitness - fitness) / max_fitness
            for fitness in current_fitnesses_sorted
        ]
        sum_probs = sum(probs_from_fitness)
        probs_from_fitness = [fitness / sum_probs for fitness in probs_from_fitness]

        for i in range(n_to_survive):
            new_generation.append(current_generation_individuals_sorted[i])

        # print(probs_from_fitness)
        for _ in range(n_to_get_by_crossover):
            parent_choices = np.random.choice(
                current_generation_individuals_sorted,
                2,
                replace=False,
                p=probs_from_fitness,
            )
            new_individual = self.crossover_two_individuals(
                parent_choices[0], parent_choices[1]
            )
            new_individual.mutate(self.mutation_rate)

            new_generation.append(new_individual)

        self.current_generation_individuals = new_generation

    def evaluate_generation_fitness(self):
        fitnesses = []
        for ind in self.current_generation_individuals:
            ind.calc_fitness(self.distance_matrix)
            fitnesses.append(ind.fitness)

        self.current_fitnesses = fitnesses

    def log_generation_stats(self):
        avg_fitness = np.mean(self.current_fitnesses)
        best_fitness, worst_fitness = (
            min(self.current_fitnesses),
            max(self.current_fitnesses),
        )

        print(
            f"generation {self.current_generation_i}\t{avg_fitness=}\n{best_fitness=}\t{worst_fitness=}"
        )

    def crossover_two_individuals(self, ind_1: GeneticSolution, ind_2: GeneticSolution):
        resulting_gene: list[int] = []
        genes_to_pick = []

        for i in range(self.n_points):
            if (
                ind_1.genes[i] not in resulting_gene
                and ind_1.genes[i] not in genes_to_pick
            ):
                genes_to_pick.append(ind_1.genes[i])

            if (
                ind_2.genes[i] not in resulting_gene
                and ind_2.genes[i] not in genes_to_pick
            ):
                genes_to_pick.append(ind_2.genes[i])

            picked_gene_id = random.randint(0, len(genes_to_pick) - 1)
            resulting_gene.append(genes_to_pick[picked_gene_id])

            del genes_to_pick[picked_gene_id]

        resulting_individual = GeneticSolution(resulting_gene)
        return resulting_individual

    def run_evolution(self):
        self.generate_initial_generation()
        self.evaluate_generation_fitness()
        self.log_generation_stats()
        self.visualize_best_individual()

        self.current_generation_i += 1

        while self.current_generation_i < self.n_generations:
            self.breed_new_generation()
            self.evaluate_generation_fitness()
            self.log_generation_stats()
            self.visualize_best_individual()

            self.current_generation_i += 1

    def visualize_best_individual(self):
        best_solution_genes = self.current_generation_individuals[
            self.current_fitnesses.index(min(self.current_fitnesses))
        ].genes
        self.visualize_map(
            os.path.join(
                self.visualization_output_dir_path,
                f"map_{self.current_generation_i}.jpg",
            ),
            best_solution_genes,
        )
