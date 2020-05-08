import numpy as np
import random
import matplotlib.pyplot
import pickle

class GA:
    """
    Genetic Algorithom
    """
    # Parameters -- GA
    population_size = None # 100
    parents_num = None # Number of solutions to be selected as parents
    max_generations = None # 150 or 200
    k_tournament = None # for the selection
    mutation_propotion = None # Percentage of genes to mutate
    #best_outputs = None
    #best_fitness_outputs = None 

    # Parameters -- information about the generations (Numpy Array)
    best_solution = [] # Best allocation solution of the each generation
    best_fitness = [] # The fitness value of the best solution for each generation
    #least_latency = [] # The smallest latency value of the best solution for each generation 


    # Parameters -- IO (not decided yet)
    task = None 
    fogcapacity = None 
    fogremain = None 
    outputs = None

    # Flags
    run_completed = False
    parameters_valid = False 

    #def __init__(self, population_size=100, parents_num=30, max_generations=150, k_tournament=3, mutation_propotion=0.05, task, fogcapacity=250.0, fogremain):

    def __init__(self, 
                population_size, 
                parents_num, 
                max_generations, 
                k_tournament, 
                mutation_propotion,
                task, 
                fogcapacity, 
                fogremain):

        if parents_num > population_size:
            print('Error: The number of parents for mating selection cannot be greater than the population size.\n')
            self.parameters_valid = False 
            return 

        # PS: There might be other parameters to be checked.
        self.parameters_valid = True 

        self.population_size = population_size
        self.parents_num = parents_num
        self.max_generations = max_generations
        self.k_tournament = k_tournament
        self.mutation_propotion = mutation_propotion
        self.task = task 
        self.fogcapacity = fogcapacity
        self.fogremain = fogremain


        #self.best_fitness = []
        self.best_solution = []
        self.least_latency = []

        self.offspring_num = self.population_size - self.parents_num

        self.initialise_population()

    def initialise_population(self):
        """
        Create an initial population. Generate a random array -> propotion of each subtask allocated to the respective fog node.
        If the sum < 1, which means there are unallocated subtasks, the propotion is allocated to the cloud.
        """
        self.population = np.zeros(shape=(self.population_size, 5))
        for i in range(self.population_size):
            a = np.random.uniform(size=4) # Default four fog nodes.
            if a.sum() > 1.0:
                a = a / (a.sum())
                chromosome = self.task * a
                solution = np.append(chromosome, 0.0)
            else:
                cloud = self.task * (1.0 - a.sum())
                chromosome = self.task * a
                solution = np.append(chromosome, cloud)
            self.population[i] = solution


    
    def run(self):
        """
        Run the genetic algorithm. 
        """
        if self.parameters_valid == False:
            print("Error: The run() method cannot be excuted with invalid parameters. \nPlease check the parameters first.\n")
            return 
        
        for g in range(self.max_generations):

            fitness = self.calculate_fitness()
            parents = self.rank_selection(fitness, self.parents_num)
            offspring_crossover = self.crossover(parents, offspring_size=(self.offspring_num, 5))
            offspring_mutation = self.mutation(offspring_crossover)
            offspring_re = self.rearrange(offspring_mutation, offspring_size=(self.offspring_num, 5))

            # Create a new generation based on the selected parents and the offspring
            self.population[0:parents.shape[0], :] = parents
            self.population[parents.shape[0]:, :] = offspring_re

        # All the run() methods are done.
        self.run_completed = True
    






    def calculate_fitness(self):
        """
        Calculate the fitness values of all the allocate solutions in the current population.
        Return an array of the calculated fitness values.
        """
        
        if self.parameters_valid == False:
            print("Error: Please check the parameters before calling calculate_fitness() function.\n")
            return []

        compute_rate = np.append(self.fogremain, 10000.0) # with each element represents the compute rate of each node. (Mbps)
        transfer_rate = np.array([1000.0, 200.0, 150.0, 250.0, 20.0]) # considering fog1 as the master node, the transfer rate between the user and fog1 should be higher.
        #perror = np.array([0.0102, 0.0241, 0.0204, 0.0173, 0.0091]) # packet error rate
        perror = np.zeros(5)
        l_para = (1.0 + perror) / (1.0 - perror)
        penalty = 1.0

        latency = self.population / transfer_rate * l_para + self.population / compute_rate
        time = latency.max(axis=1)

        tj1 = np.where(self.population>=0, 0, -self.population)
        tj2 = np.abs((np.sum(self.population, axis=1) - self.task))
        tj = np.sum(tj1) + tj2  

        fit = time + tj * penalty 
        fitness = np.where(fit != 0, 1.0 / fit, -100.0)
        #fitness = 1.0 / fit 

        
        self.best_fitness.append(np.max(fitness))
        #self.least_latency.append(np.max(1.0 / fitness))
        self.best_solution.append(self.population[np.where(fitness == np.max(fitness))])

        return fitness 
    
    def selection(self, fitness, parents_num):
        """
        default: k-tournament, where k = 3
        """
        parents = np.zeros((parents_num, 5))
        for num in range(parents_num):
            rand_indices = np.random.randint(low=0.0, high=len(fitness), size=self.k_tournament)
            k_group = fitness[rand_indices]
            selected_parent_idx = np.where(k_group == np.max(k_group))[0][0]
            parents[num, :] = self.population[rand_indices[selected_parent_idx], :]
        return parents 

    def rank_selection(self, fitness, parents_num):
        """
        an alternative selection method.
        Selecting the best individuals in the current generation as parents.
        Compare this method with k_tournament later.
        """
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        parents = np.zeros((parents_num, self.population.shape[1]))
        for num in range(parents_num):
            parents[num, :] = self.population[fitness_sorted[num], :]
            return parents 


    def crossover(self, parents, offspring_size):
        """
        crossover type - mathmetical crossover.
         """
        offspring = np.zeros(offspring_size)
        

        for k in range(0, offspring_size[0], 2):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k+1) % parents.shape[0]
            lam = np.random.uniform()
            offspring[k] = lam * parents[parent1_idx] + (1.0-lam) * parents[parent2_idx]
            offspring[k+1] = (1.0-lam) * parents[parent1_idx] + lam * parents[parent2_idx]

            # I intended to practicalise/feasiblise the infeasible solutions. 
            # might add this part later.

        return offspring 

    def mutation(self, offspring):
        """
        random mutation. 
        The ultimate mutation type -- to be decided.
        """
        mutation_indices = np.array(random.sample(range(0, offspring.shape[0]), int(self.offspring_num * self.mutation_propotion)))
        for idx in mutation_indices:
            random_value = np.random.uniform(0, self.task / 2, 1)
            mutation_gene = np.array(random.sample(range(0, offspring.shape[1]), 1))
            offspring[idx, mutation_gene] = random_value
        return offspring


    def rearrange(self, offspring, offspring_size):
        """
        Rearrange to make the solutions in infeasible areas feasible.
        """
        re_offspring = np.zeros(offspring_size)
        reo = np.where(offspring<0, -offspring, offspring)
        for r in range(offspring_size[0]):
            if np.sum(reo[r]) > self.task:
                re_offspring[r] = reo[r] * np.sum(reo[r]) / self.task
            elif np.sum(reo[r]) < self.task:
                reo[r][4] = self.task - (np.sum(reo[r]) - reo[r][4])
                re_offspring[r] = reo[r]
        return re_offspring
            


    def best_outputs(self):
        """
        Return the best outputs (solutoins and fitness) of the last population.
        (If the run() function is not called, it returns 2 empty lists.)
        """

        if self.run_completed == False:
            print("Warning: Calling the best_outputs() function failed. \nThe run() is not yet called and thus Genetic algorithm did not evolve the solution. Thus, the best solution is retrieved from the initial population without being evolved.\n")
            return [], []
        
        last_fitnesses = self.calculate_fitness()
        best_index = np.where(last_fitnesses == np.max(last_fitnesses))
        best_outputs = self.population[best_index, :][0][0]
        best_outputs_fitness = last_fitnesses[best_index][0]

        return best_outputs, best_outputs_fitness

    def plot_result(self):
        """
        Creating 2 plots that summarizes how the solutions evolved.
        The first plot is between the iteration number and the function output based on the current parameters for the best solution.
        The second plot is between the iteration number and the fitness value of the best solution.
        """
        if self.run_completed == False:
            print("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        #matplotlib.pyplot.figure()
        #matplotlib.pyplot.plot(self.least_latency)
        #matplotlib.pyplot.xlabel("Iteration")
        #matplotlib.pyplot.ylabel("Latency")
        #matplotlib.pyplot.show()

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.best_fitness)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Fitness")
        matplotlib.pyplot.show()

    def save(self, filename):
        
        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)

def load(filename):

    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = pickle.load(file)
    except (FileNotFoundError):
        print("Error loading the file. Please check if the file exists.")
    return ga_in
