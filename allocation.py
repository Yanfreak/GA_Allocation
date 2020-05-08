import GeneticA
import numpy as np 


# generate a random number -- task load. (Mb)
task = 350
#task = np.random.uniform(250, 500)


fogcapacity = 300

# fogremain = np.random.uniform(0, fogcapacity, 4)

max_generations = 100 
population_size = 200 
parents_num = 40 
k_tournament = 3

# Parameters of the mutation operation.
mutation_propotion = 0.1 

# non_optimised = task / fogremain[0] + task / 10206.1022


# c = np.random.uniform(size=4) # Default four fog nodes.


#if c.sum() > 1.0:
#    c = c / (c.sum())
#    b = task * c
#    s = np.append(b, 0.0)
#else:
#    d = task * (1.0 - c.sum())
#    b = task * c
#    s = np.append(b, d)
##print(s)

# compute_rate1 = np.append(fogremain, 10000.0) # with each element represents the compute rate of each node. (Mbps)
transfer_rate1 = np.array([1000.0, 200.0, 150.0, 250.0, 20.0]) # considering fog1 as the master node, the transfer rate between the user and fog1 should be higher.
#perror1 = np.array([0.0102, 0.0241, 0.0204, 0.0173, 0.0091]) # packet error rate
perror1 = np.zeros(5)
l_para1 = (1.0 + perror1) / (1.0 - perror1)


# latency = s / transfer_rate1 * l_para1 + s / compute_rate1
# time_un = np.max(latency)

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.


repeat = 50

time_for_average = np.zeros(repeat)
time_for_non = np.zeros(repeat)
time_for_un = np.zeros(repeat)


for i in range(repeat):
    fogremain = np.random.uniform(0, fogcapacity, 4)
    fogremain[0] = 150.0
    GeneticA_instance = GeneticA.GA(population_size=population_size, 
                                parents_num=parents_num, 
                                max_generations=max_generations, 
                                k_tournament=k_tournament,
                                mutation_propotion=mutation_propotion, 
                                task=task, 
                                fogcapacity=fogcapacity, 
                                fogremain=fogremain)
    time_for_non[i] = task / fogremain[0] + task / 10206.1022
    c = np.random.uniform(size=4) # Default four fog nodes.
    if c.sum() > 1.0:
        c = c / (c.sum())
        b = task * c
        s = np.append(b, 0.0)
    else:
        d = task * (1.0 - c.sum())
        b = task * c
        s = np.append(b, d)
    compute_rate1 = np.append(fogremain, 10000.0)
    latency = s / transfer_rate1 * l_para1 + s / compute_rate1
    time_for_un[i] = np.max(latency)

    GeneticA_instance.run()
    best_result, best_result_fitness = GeneticA_instance.best_outputs()
    time_for_average[i] = 1.0 / best_result_fitness

#GeneticA_instance.run()

#GeneticA_instance.plot_result()

best_result, best_result_fitness = GeneticA_instance.best_outputs()
print("task:", task)
# print("fog remain:", fogremain)
print("non_optimised", np.average(time_for_non))
print("Time_un:", np.average(time_for_un)) # random arrange the task
# print("The allocation: ", best_result)
# print("The fitness value: ", best_result_fitness)
print("Time: ", np.average(time_for_average))

filename = 'ga1'
GeneticA_instance.save(filename=filename)

loaded_GeneticA_instance = GeneticA.load(filename=filename)
# loaded_GeneticA_instance.plot_result()
# print(loaded_GeneticA_instance.best_outputs())
