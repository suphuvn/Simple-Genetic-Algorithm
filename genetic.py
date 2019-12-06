'''
README
1. set params in params_dict
2. set objective functions name in the entrance of program (line 284)
3. run 'python3 genetic.py'

GramacyLee        1D  http://benchmarkfcns.xyz/benchmarkfcns/gramacyleefcn.html
Beale             2D  http://benchmarkfcns.xyz/benchmarkfcns/bealefcn.html
GoldsteinPrice    2D  http://benchmarkfcns.xyz/benchmarkfcns/goldsteinpricefcn.html
Himmelblau        2D  http://benchmarkfcns.xyz/benchmarkfcns/himmelblaufcn.html
DeJong            nD  http://benchmarkfcns.xyz/benchmarkfcns/spherefcn.html
Rosenbrock        nD  http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html
Rastrigin         nD  http://benchmarkfcns.xyz/benchmarkfcns/rastriginfcn.html
Test25-51         1D  http://www.cargo.wlu.ca/OFs/2ccABvarsFormat/

n                 the size of the population, the number of chromosomes/individuals
value_num         the number of values each chromosome contains
value_len         the length each value is encoded
pc                probability of crossover on two values
pm                probability of mutation on each bit of value
iteration         number of terms
is_max            True -> find maximum of OF, False -> find minimum of OF
'''
from __future__ import division
import random
from random import randint
import math
import time
import re
from itertools import combinations


# key: objective function name
# value: [n, value_num, value_len, pc, pm, iteration, is_max]
param_dict = {
    'GramacyLee': [100, 1, 10, 0.5, 0.005, 100, False],
    'Beale': [100, 2, 10, 0.8, 0.01, 500, False],
    'GoldsteinPrice': [100, 2, 10, 0.8, 0.005, 100, False],
    'Himmelblau': [100, 2, 10, 0.8, 0.01, 20, False],
    'DeJong': [100, 10, 10, 1.0, 0.002, 1000, False],
    'Rosenbrock': [100, 7, 10, 1.0, 0.05, 1000, False],
    'Rastrigin': [100, 7, 10, 1.0, 0.02, 1000, False],
    'Test25': [100, 1, 25, 0.8, 0.01, 50, False],
    'Test27': [100, 1, 27, 0.8, 0.01, 54, False],
    'Test29': [100, 1, 29, 0.8, 0.01, 58, False],
    'Test31': [100, 1, 31, 0.8, 0.01, 62, False],
    'Test33': [100, 1, 33, 0.8, 0.01, 66, False],
    'Test35': [100, 1, 35, 0.8, 0.01, 70, False],
    'Test37': [100, 1, 37, 0.8, 0.01, 74, False],
    'Test39': [100, 1, 39, 0.8, 0.01, 78, False],
    'Test41': [100, 1, 41, 0.8, 0.01, 82, False],
    'Test43': [100, 1, 43, 0.8, 0.01, 86, False],
    'Test45': [100, 1, 45, 0.8, 0.01, 90, False],
    'Test47': [100, 1, 47, 0.8, 0.01, 94, False],
    'Test49': [100, 1, 49, 0.8, 0.01, 98, False],
    'Test51': [100, 1, 51, 0.8, 0.01, 102, False],
}


class Chromosome:
    def __init__(self, string):
        self.string = string
        self.x = self.get_x()
        self.y = self.get_y()
        self.fitness = self.get_fitness()
        self.p = 0

    # randomly generate input x of the objective function
    # each value takes value_len binary bits, thus can represent a integer among 0 - 2^value_len
    # scale the range 0 - 2^value_len to fit the given objective function's domain
    def get_x(self):
        if of == 'GramacyLee':
            # x         domain: 0.5 <= x <= 2.5, not starts from -0.5
            return (int(self.string, 2) + 256) / 512
        elif of == 'Beale':
            # x1 x2     domain: -4.5 <= x1, x2 <= 4.5
            return [(int(self.string[i:i + value_len], 2) - 512) / 114
                    for i in range(0, len(self.string), value_len)]
        elif of == 'GoldsteinPrice':
            # x1 x2     domain: -2 <= x1, x2 <= 2
            return [(int(self.string[i:i + value_len], 2) - 512) / 256 for i in range(0, len(self.string), value_len)]
        elif of == 'Himmelblau':
            # x1 x2     domain: -4 <= x1, x2 <= 4
            return [(int(self.string[i:i + value_len], 2) - 512) / 128 for i in range(0, len(self.string), value_len)]
        elif of in ['DeJong', 'Rastrigin']:
            # x[]       domain: -5.12 <= xi < 5.12
            return [(int(self.string[i:i + value_len], 2) - 512) / 100 for i in range(0, len(self.string), value_len)]
        elif of in ['Rosenbrock']:
            # x[]       domain: -2.048 <= xi <= 2.048
            return [(int(self.string[i:i + value_len], 2) - 512) / 250 for i in range(0, len(self.string), value_len)]
        elif re.match('Test*', of):
            return [-1 if elem == '0' else 1 for elem in self.string]

    # calculate output y of given objective function by input x
    def get_y(self):
        if of == 'GramacyLee':
            # subtract the minimum at the end so that the new minimum is 0
            return (self.x - 1) ** 4 + math.sin(10 * math.pi * self.x) / (2 * self.x) - (-0.869011134989500)
        elif of == 'Beale':
            return (1.5 - self.x[0] + self.x[0] * self.x[1]) ** 2 \
                   + (2.25 - self.x[0] + self.x[0] * (self.x[1] ** 2)) ** 2 \
                   + (2.625 - self.x[0] + self.x[0] * (self.x[1] ** 3)) ** 2
        elif of == 'GoldsteinPrice':
            # subtract the minimum at the end so that the new minimum is 0
            return (1 + (self.x[0] + self.x[1] + 1) ** 2
                    * (19 - 14 * self.x[0] + 3 * self.x[0] ** 2
                       - 14 * self.x[1] + 6 * self.x[0] * self.x[1] + 3 * self.x[1] ** 2)) \
                   * (30 + (2 * self.x[0] - 3 * self.x[1]) ** 2
                      * (18 - 32 * self.x[0] + 12 * self.x[0] ** 2 + 48 * self.x[1] - 36 * self.x[0] * self.x[1] + 27 *
                         self.x[1] ** 2)) \
                   - 3
        elif of == 'Himmelblau':
            return (self.x[0] ** 2 + self.x[1] - 11) ** 2 + (self.x[0] + self.x[1] ** 2 - 7) ** 2
        elif of == 'DeJong':
            return sum(elem ** 2 for elem in self.x)
        elif of == 'Rosenbrock':
            return sum(100 * ((self.x[i + 1] - (self.x[i] ** 2)) ** 2) + (1 - self.x[i]) ** 2
                       for i in range(1, len(self.x) - 1))
        elif of == 'Rastrigin':
            return 10 * len(self.x) + sum(elem ** 2 - 10 * math.cos(2 * math.pi * elem) for elem in self.x)
        elif re.match('Test*', of):
            return abs(sum(elem[0] * elem[1] for elem in list(combinations(self.x, 2))))

    # generate fitness that we want to maximize
    # fitness == y if the aim is to maximize the objective function
    # fitness == 1 / (1 + y) if the aim is to minimize the objective function
    def get_fitness(self):
        if is_max:
            return self.y
        else:
            return 1 / (1 + self.y)

    def calc_p(self, sum_fitness):
        self.p = self.fitness / sum_fitness

    # customize printing format, called by print()
    def __repr__(self):
        return '%r\tx:%r\ty:%r\tfitness:%r\n' % (self.string, self.x, self.y, self.fitness)


# randomly generate a population of size n
def generate_population():
    new_population = list()
    for _ in range(n):
        str_list = [str(randint(0, 1)) for _ in range(value_num * value_len)]
        chromosome = Chromosome(''.join(str_list))
        new_population.append(chromosome)
    # print('population:\n%r\n' % new_population)
    return new_population


def reproduction():
    fitness_list = list()
    for i in range(n):
        fitness_list.append(population[i].fitness)

    # find two chromosomes with the largest fitness and put them at the beginning
    elite1_fitness = max(fitness_list)
    elite1_index = fitness_list.index(elite1_fitness)
    fitness_list.pop(elite1_index)
    elite2_fitness = max(fitness_list)
    elite2_index = fitness_list.index(elite2_fitness)

    elite1 = population[elite1_index]
    elite2 = population[elite2_index]
    population[elite1_index] = population[0]
    population[elite2_index] = population[1]
    population[0] = elite1
    population[1] = elite2

    # calculate p for non-elites chromosome
    sum_fitness = 0.0
    for i in range(2, n):
        sum_fitness += population[i].fitness
    print(sum_fitness)
    for i in range(2, n):
        population[i].calc_p(sum_fitness)

    # use roulette to select string
    # roulette contains n-2 values
    p = 0
    roulette = list()
    for i in range(2, n):
        p += population[i].p
        roulette.append(p)
    # print('roulette:\n%r\n' % roulette)

    new_population = list()
    # 2 elites reproduce directly, do not go through roulette
    new_population.append(population[0])
    new_population.append(population[1])
    # n-2 left chromosomes go through roulette
    for i in range(n - 2):
        p = random.random()
        for j in range(n - 2):
            if p <= roulette[j]:
                new_population.append(population[j + 2])
                break
    # print('after reproduction:\n%r\n' % new_population)
    return new_population


def crossover():
    new_population = list()
    # 2 elites pass directly to next generation, do not go through crossover
    new_population.append(population[0])
    new_population.append(population[1])
    # n-2 left chromosomes go through crossover at probability pc
    for i in range(1, n // 2):
        father = population[i * 2]
        mother = population[i * 2 + 1]
        brother = father
        sister = mother
        for j in range(value_num):
            # has a probability of pc to do crossover
            if random.random() <= pc:
                pos = j * value_len + randint(1, value_len - 1)
                brother = Chromosome(father.string[:pos] + mother.string[pos:])
                sister = Chromosome(father.string[:pos] + mother.string[pos:])
                father = brother
                mother = sister
        new_population.append(brother)
        new_population.append(sister)
    # print('\nafter crossover:\n%r\n' % new_population)
    return new_population


def mutation():
    # n-2 left chromosomes go through mutation at probability pm
    for i in range(2, n):
        for j in range(value_num):
            for k in range(value_len):
                # has a probability of p to do mutation
                if random.random() <= pm:
                    string = population[i].string
                    flip = lambda x: '1' if x is '0' else '0'
                    char = string[j * value_len + k]
                    char = flip(char)
                    new_string = string[:j * value_len + k] + char + string[j * value_len + k + 1:]
                    # print('replace %r with %r' % (string, new_string))
                    population[i] = Chromosome(new_string)
    # print('\nafter mutation:\n%r\n' % population)
    return population


def get_record():
    index = -1
    max_fitness = -1
    sum_y = 0
    sum_fitness = 0
    for i in range(n):
        sum_y += population[i].y
        sum_fitness += population[i].fitness
        if population[i].fitness > max_fitness:
            max_fitness = population[i].fitness
            index = i
    return population[index].y, sum_y / n, population[index].fitness, sum_fitness / n


def draw_plot(data):
    sns.set()
    plt.title('GA on %s function  is_max=%r\nn=%r  value_num=%r  value_len=%r\n'
              'p_crossover=%r  p_mutation=%r running_time=%0.2fs'
              % (of, is_max, n, value_num, value_len, pc, pm, running_time))
    plt.plot(data['optimum_y'], color='C1', linestyle='-', linewidth=2, label='optimum_y')
    plt.plot(data['avg_y'], color='C1', linestyle='-', linewidth=0.5, label='avg_y')
    plt.xlabel('iteration')
    plt.ylabel('y', color='C1')
    plt.legend(loc=(.2, .6), frameon=False)
    right_axis = plt.twinx()
    right_axis.plot(data['max_fitness'], color='C2', alpha=1, linestyle='-.', linewidth=2, label='max_fitness')
    right_axis.plot(data['avg_fitness'], color='C2', alpha=1, linestyle='-.', linewidth=0.5, label='avg_fitness')
    right_axis.set_ylabel('fitness', color='C2')
    right_axis.legend(loc=(.6, .6), frameon=False)
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    # choose an objective function from
    # 'GramacyLee' 'Beale' 'GoldsteinPrice' 'Himmelblau' 'DeJong' 'Rosenbrock' 'Rastrigin' or 'Test25' to 'Test51'
    of = 'Test25'
    n = param_dict[of][0]
    value_num = param_dict[of][1]
    value_len = param_dict[of][2]
    pc = param_dict[of][3]
    pm = param_dict[of][4]
    iteration = param_dict[of][5]
    is_max = param_dict[of][6]

    population = generate_population()
    keys = {'optimum_y', 'avg_y', 'max_fitness', 'avg_fitness'}
    record_dict = dict([(key, []) for key in keys])
    iterate = 0
    while iterate < iteration:
        population = reproduction()
        population = crossover()
        population = mutation()
        optimum_y, avg_y, max_fitness, avg_fitness = get_record()
        record_dict['optimum_y'].append(optimum_y)
        record_dict['avg_y'].append(avg_y)
        record_dict['max_fitness'].append(max_fitness)
        record_dict['avg_fitness'].append(avg_fitness)
        iterate += 1
    print('after genetic algoritm:\n%r\n' % population)
    running_time = time.time() - start_time
    print(running_time)
    draw_plot(record_dict)