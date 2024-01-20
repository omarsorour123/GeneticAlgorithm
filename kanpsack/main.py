import random
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, n=7, arr=None, mutate_prob=0.05):
        self.n = n #len of items
        if arr == None: #create random Individual
            self.bits = [random.choice([0, 1]) for _ in range(self.n)]
        else: #try to mutate Individual
            self.bits = arr
            if mutate_prob > np.random.rand():
                mutate_index = np.random.randint(self.n - 1)
                self.bits[mutate_index] ^= 1
    def fitness(self, items, capacity):
        weight,value=0,0
        for i in range(self.n):
            if self.bits[i]==1:
                value += items[i][0] #add value
                weight += items[i][1] #add weight
        # if weight of Individual met capacity constrains
        # return the value
        return value if weight <= capacity else 0

class Population:
    def __init__(self,n=7,size=10,elitism=0.1 ,mutate_prob=0.05,random_retain=0.1):
        self.size=size
        self.elitism=elitism
        self.mutate_prob=mutate_prob
        self.random_retain=random_retain
        self.n=n
        self.pop = [Individual(n=n,mutate_prob=mutate_prob) for _ in range(size)]

    def bestIndividual(self,items,capacity):
        best=0
        for i in range(self.size):
            fit = self.pop[i].fitness(items,capacity)
            if best < fit:
                best = fit
                bits = self.pop[i].bits
        return bits
    def grade(self,items,capacity):
        grade=[]
        for i in range(self.size):
            grade.append(self.pop[i].fitness())
        return sorted(grade,reverse=True)
    def tournmentSelection(self,items,Capacity,n_participant=2):
        tournment_candidites = random.sample(self.pop,n_participant)
        winner = max(tournment_candidites,key=lambda x:x.fitness(items=items,capacity=capacity))
        return winner

    def rouletteWheelSelection(self, items, capacity):
        total_fitness = sum(individual.fitness(items, capacity) for individual in self.pop)

        if total_fitness == 0:
            # If total fitness is zero, return a random individual
            return random.choice(self.pop)
        else:
            selected = random.choices(self.pop,
                                      weights=[individual.fitness(items,
                                                                  capacity) / total_fitness if total_fitness > 0 else 0
                                               for individual in self.pop],
                                      k=1)
            return selected[0]
    def getParent(self,items,capacity):
        prob = random.random()
        if prob>0.5:
            return self.tournmentSelection(items,capacity)
        else:
            return self.rouletteWheelSelection(items, capacity)

    def crossover_mutation(self, selected):
        offspring = []

        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]  # Wrap around if odd number of parents

            crossover_point = random.randint(1, self.n - 1)
            child1_bits = parent1.bits[:crossover_point] + parent2.bits[crossover_point:]
            child2_bits = parent2.bits[:crossover_point] + parent1.bits[crossover_point:]

            child1 = Individual(n=self.n, arr=child1_bits, mutate_prob=self.mutate_prob)
            child2 = Individual(n=self.n, arr=child2_bits, mutate_prob=self.mutate_prob)

            offspring.extend([child1, child2])

        return offspring
    def evolve(self, items, capacity, generations=50):
        fitness_evolution = []

        for generation in range(generations):
            selected_parents = [self.getParent(items, capacity) for _ in range(self.size)]
            offspring = self.crossover_mutation(selected_parents)

            combined_pop = self.pop + offspring

            # Apply elitism
            elite_count = int(self.size * self.elitism)
            next_generation = sorted(combined_pop, key=lambda x: x.fitness(items, capacity), reverse=True)[:elite_count]

            # Randomly retain individuals
            random_count = int(self.size * self.random_retain)
            next_generation += random.sample(combined_pop, random_count)

            # Generate new individuals
            remaining_count = self.size - len(next_generation)
            next_generation += [Individual(n=self.n, mutate_prob=self.mutate_prob) for _ in range(remaining_count)]

            self.pop = next_generation

            # Record the best fitness value in this generation
            best_fitness = max(self.pop, key=lambda x: x.fitness(items, capacity)).fitness(items, capacity)
            fitness_evolution.append(best_fitness)

        return fitness_evolution







items = [
    (45, 30), (60, 40), (10, 8), (32, 20), (15, 10), (22, 15), (18, 12),
    (50, 35), (40, 25), (28, 18), (12, 7), (35, 22), (24, 16), (30, 20),
    (55, 38), (42, 27), (20, 15), (48, 32), (38, 24), (25, 16), (33, 21),
    (15, 10), (28, 18), (36, 23),(5,5)
]

capacity = 100
pop = Population(n=25)

# Run the evolution for a certain number of generations
generations = 15
fitness_evolution = pop.evolve(items, capacity, generations)

# Plot the evolution of the fitness function
plt.plot(range(1, generations + 1), fitness_evolution, marker='o')
plt.title('Evolution of Fitness Function')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.show()
