from collections import defaultdict
import operator
import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle


def choromosome_validity(distance_matrix, path_list):
    costMatrix = np.array(distance_matrix)
    for i in range(0, len(path_list) - 1):
        city_a = path_list[i]
        city_b = path_list[i + 1]
        if costMatrix[city_a - 1][city_b - 1] == 0:
            return False
    return True


def get_cities(filename):
    file = open(filename)
    data = file.readlines()
    cities = []
    for i in data:
        cities.append(i.strip('\n'))
    return cities


def nxn_distance_matrix(filename, total_cities):
    file = open(filename)
    data = file.readlines()
    row_counter = 0
    distance_matrix = np.zeros(shape=(total_cities, total_cities))
    for i in data:
        i = i.strip("\n")
        i = i.split(" ")
        for j in range(total_cities):
            distance_matrix[row_counter][j] = i[j]
        row_counter += 1

    return distance_matrix


def get_starting_and_other_points(cities):
    initial = [1]
    other = []
    for i in range(2, len(cities) + 1):
        other.append(i)
    return (initial, other)


def get_choromosome(initial, other, distance_matrix):
    isvalid = False
    chromosome = []
    while isvalid == False:
        random_arrangement = list(np.random.permutation(other))
        chromosome = initial + random_arrangement
        if choromosome_validity(distance_matrix, chromosome) == True:
            isvalid = True
    return chromosome


def get_initial_population(population_size, initial, other, distance_matrix):
    pop = []
    for i in range(population_size):
        pop.append(get_choromosome(initial, other, distance_matrix))
    return pop

def calculateFitness(distances, choromosome):
    distance_matrix = np.array(distances)
    fitness = 0
    for i in range(len(choromosome)-1):
        city_a = choromosome[i]
        city_b = choromosome[i + 1]
        fitness = fitness + distance_matrix[city_a - 1][city_b - 1]
    return -(fitness)


def myown_crossover(ch1, ch2):  #   first i apply one point crossover , then i created a list of number of cities and then i shuffled it. after then i compared it with the offsprings and removed the repeated values in them
    point = len(ch1) - 3
    el = list(range(1, len(ch1)+1))
    offspring1 = ch1[:point] + ch2[point:]
    offspring2 = ch2[:point] + ch1[point:]
    shuffle(el)
    for i in el:
        if i not in offspring1:
            offspring1.append(i)

    shuffle(el)
    for i in el:
        if i not in offspring2:
            offspring2.append(i)

        # mutation
    first = list(set(offspring1))
    second = list(set(offspring2))
    randomNumber = random.randrange(1, 10)
    if randomNumber == 5:
        temp = first[3]
        first[3] = first[5]
        first[5] = temp

    pop = []
    pop.append(first)
    pop.append(second)
    return pop

def uniform_crossover(ch1, ch2):
    offspring1 = []
    offspring2 = []

    p = [ch1,ch2]
    p1 = [ch2,ch1]
    for i in range(len(p[0])):
        if p[1][i] in p[0]:  # cannot risk crossover, keep basic gene
            offspring1.append(p[0][i])
        else:  # standard uniform crossover
            offspring1.append(p[random.randint(0, 1)][i])


    for i in range(len(p1[0])):
        if p1[1][i] in p1[0]:  # cannot risk crossover, keep basic gene
            offspring2.append(p1[0][i])
        else:  # standard uniform crossover
            offspring2.append(p1[random.randint(0, 1)][i])


    # mutation
    randomNumber = random.randrange(1, 10)
    if randomNumber == 5:
        temp = offspring1[3]
        offspring1[3] = offspring1[5]
        offspring1[5] = temp

    pop = []
    pop.append(offspring1)
    pop.append(offspring2)
    return pop


def stocastic_selection(population,population_size,distances):
    fitnessDict = defaultdict(list)
    for i in population:
        fitnessDict[calculateFitness(distances,i)] = i

    sortedFitness = list(reversed(sorted(fitnessDict.items(), key=operator.itemgetter(0))))
    population_new = []
    pop = 0
    for i in range(population_size):
        population_new.append(sortedFitness[pop][1])
        pop+=1

    return population_new


def tournament_selection(population,population_size,distances,tournament_size):
    fitnessDict = defaultdict(list)
    for i in population:
        fitnessDict[calculateFitness(distances,i)] = i

    sortedFitness = list(fitnessDict.items())
    new_list = random.sample(sortedFitness, tournament_size)
    population_new = []
    pop = 0
    for i in range(population_size):
        population_new.append(new_list[pop][1])
        pop+=1

    return population_new




def genetic_algorithm(initial_population_size , crossover , selection ,tournament_size = 4):
    cities = get_cities("city_names.txt")
    (starting_point, remaining_points) = get_starting_and_other_points(cities)
    distance_matrix = nxn_distance_matrix("city_distance_matrix.txt", len(cities))

    population = get_initial_population(initial_population_size, starting_point, remaining_points, distance_matrix)

    fitnessDict = defaultdict(list)
    for i in range(initial_population_size):
        fitnessDict[calculateFitness(distance_matrix, population[i])] = population[i]

    for i in range(0,10000):
        clone_for_population = population[:]
        for i in range(0,len(clone_for_population),2):
            if crossover == "uniform":
                pop_intermediate = uniform_crossover(clone_for_population[i], clone_for_population[i+1])    ## choose crossover method here
            else:
                pop_intermediate = myown_crossover(clone_for_population[i], clone_for_population[i+1])    ## choose crossover method here
            population.extend(pop_intermediate)
        clone_for_population[:] = []
        if selection == "stocastic":
            population = stocastic_selection(population, initial_population_size, distance_matrix)  ## choose selection method here
        else:
            population = tournament_selection(population, initial_population_size,distance_matrix,tournament_size)  ## choose selection method here
        for i in range(initial_population_size):
            fitnessDict[calculateFitness(distance_matrix, population[i])] = population[i]

    return fitnessDict,cities

def get_lat_long(filename):
    file = open(filename)
    data = file.readlines()
    row_counter = 0
    lat = []
    long = []
    for i in data:
        i = i.strip("\n")
        i = i.split(" ")
        lat.append(i[0])
        long.append(i[1])
        row_counter += 1

    return (lat,long)


(fitnessDict,cities) = genetic_algorithm(4,"myown","tournament")
sortedFitness = list(reversed(sorted(fitnessDict.items(), key=operator.itemgetter(0))))
print("Fitness for the best path : " + str(sortedFitness[0][0]))
path = sortedFitness[0][1]
city_wise_path = []
for i in path:
    city_wise_path.append(cities[i-1])
print("Cities in the best path : " + str(city_wise_path))

## plotting graph for cities just for demonstration purposes.
lat,long = get_lat_long("city_coordinates.txt")
fig, ax = plt.subplots()
plt.title("Cities Plotted on Scatter Plot")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
ax.scatter(long, lat)


for i, txt in enumerate(cities):
    ax.annotate(txt, (long[i],lat[i]))
# plt.show()
plt.savefig("Initial_Representation.png")

mapping = {}
for i in range(len(cities)):
    mapping[cities[i]] = (lat[i],long[i])
lat1=[]
long1=[]
for i in range(len(city_wise_path)):
    (la,lo) = mapping[city_wise_path[i]]
    lat1.append(la)
    long1.append(lo)

## plotting graph for cities just for demonstration purposes.
fig1, ax = plt.subplots()
plt.title("Shortest Path on Scatter Plot")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
ax.scatter(long1, lat1)
plt.plot(long1, lat1, '-o')

for i, txt in enumerate(cities):
    ax.annotate(txt, (long[i],lat[i]))
# plt.show()
plt.savefig("Final_Representat2ion.png")
