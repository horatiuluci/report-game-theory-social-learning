# Horatiu Luci
# 

import numpy as np
import random
import math
import matplotlib.pyplot as plt


def get_state(population):
    s = 0
    for i in population:
        if i == 0:
            s += 1
    return s

def play_game(strategy1, strategy2, A):
    return A[strategy1][strategy2], A[strategy2][strategy1]


def select_random_with_replacement(population, n):
    n = 2
    ch1, ch2 = random.sample(range(0,50), 2)
    return ch1, ch2

def prob_imitation(beta, fitness):
    return 1/(1+math.e**(beta*(fitness[0]-fitness[1])))


def moran_step(beta, population, A, Z, mu):

    '''
    This function implements a birth-death process over the population. At time t, two players are randomly selected from the population
    '''
    fitness = [0, 0]
    selected=select_random_with_replacement(population, 2)
    for i, player in enumerate(selected):
        for j in range(len(population)):
            if j == player: continue
            players_payoffs = play_game(int(population[player]), int(population[j]), A)
            fitness[i] += players_payoffs[0]
    fitness[0],fitness[1] = fitness[0] / (Z-1), fitness[1] / (Z-1)
    if np.random.rand() < prob_imitation(beta, fitness):
        if np.random.rand() < mu:
            population[selected[0]] = np.random.randint(0,2)
        else:
            population[selected[0]] = population[selected[1]]
    else:
        if np.random.rand() < mu:
            population[selected[1]] = np.random.randint(0,2)
        else:
            population[selected[1]] = population[selected[0]]

    return population



def run_moran(beta, population, A, Z, mu):
    transitory = 1000
    generations = 100000
    nb_runs = 10

    states = [0 for i in range(Z+1)]

    for j in range(0, nb_runs):
        for i in range(0,transitory):
            population = moran_step(beta, population, A, Z, mu)
        for i in range(0,generations):
            population = moran_step(beta, population, A, Z, mu)
            states[get_state(population)]+=1
    return [states[i]/(generations*nb_runs) for i in range (0,len(states))]


def main():
    beta = 10
    mu = 0.001
    Z = 50
    V = 2
    D = 3
    T = 1
    A = np.array([
    [(V-D)/2, V],
    [0, V/2-T]
    ])


    population = np.random.rand(Z)
    population = np.rint(population)
    # print(moran_step(beta, population, A, Z, mu))

    vals = run_moran(beta, population, A, Z, mu)
    x = [i for i in range (0,51)]

    plt.plot(x, vals, 'r')
    plt.xlabel('x')
    plt.ylabel('vals')
    plt.show()




if (__name__ == '__main__'):
    main()
