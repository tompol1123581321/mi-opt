import math
import random
import matplotlib.pyplot as plt
import time
import statistics

all_searches_arrays = []
all_searches_arrays_best_values = []
cumulative_comp_list = []

mini1: int
mini2: int

"""             
DISCLAIMER:     
this project was done by multiple terrestrials:
Tomáš Polívka                       
Lukáš Gazdík                                  
Marek Trefil
Dominik Vodička
"""

def generate_random_numbers(dim, ran):
    result = []
    for index in range(dim):
        result.append(random.uniform(-ran, ran))
    return result


def evaluate_de_jong(listo):
    result = 0
    for arg in listo:
        result += math.pow(arg, 2)
    return result


def evaluate_de_jong_sec(listo):
    result = 0
    args = listo
    for index in range(len(args) - 1):
        curr_value = args[index]
        next_value = args[index + 1]
        result += 100 * \
                  math.pow((math.pow(curr_value, 2) - next_value), 2) + \
                  math.pow((1 - curr_value), 2)
    return result


def evaluate_schwefel(listo):
    result = 0
    for i in range(len(listo)):
        result += -listo[i] * math.sin(math.sqrt(abs(listo[i])))
    return 418.9829 * len(listo) - result


def random_searcher(func, dim, ran, search, ite):
    cf_results_list = []
    cf_result = search
    for i in range(ite):
        listo = generate_random_numbers(dim, ran)
        res = func(listo)
        if res < cf_result:
            cf_results_list.append(res)
            cf_result = res
        else:
            cf_results_list.append(cf_result)
    all_searches_arrays.append(cf_results_list)
    all_searches_arrays_best_values.append(cf_result)


def simulated_annealinger(func, dim, ran, search, ite):
    """Peforms simulated annealing to find a solution
        best params for each function (as far as tested):
        dejong1 5D, alpha:0.97 init_temp:800
        dejong1 10D, alpha:0.96 init_temp:800
        dejong2 5D, alpha:0.97 init_temp:800
        dejong2 10D, alpha:0.97 init_temp:800
        swaf 5D, alpha:0.99 init_temp:1000
        swaf 10D, alpha:0.99 init_temp:1000
    """

    fes_count = ite
    neighbor_call_count = 10
    alpha = 0.99
    initial_temp = 1000
    final_temp = 0.01
    current_temp = initial_temp

    cf_results_list = []
    evolved_values = []

    listo = generate_random_numbers(dim, ran)
    current_state = listo
    solution_value = func(current_state)
    best_solution_value = search
    evolved_values.append(best_solution_value)

    while fes_count > 0:
        for _ in range(neighbor_call_count):
            fes_count -= 1
            neighbor = get_neighbors(current_state, ran)
            neighbor_value = func(neighbor)
            cost_diff = solution_value - neighbor_value

            if cost_diff > 0:
                current_state = neighbor
                solution_value = neighbor_value
                if solution_value < best_solution_value:
                    best_solution_value = solution_value
            else:
                probab_that_we_accept_worse_solution = math.exp(cost_diff / current_temp)
                if random.uniform(0, 1) < probab_that_we_accept_worse_solution:
                    current_state = neighbor
                    solution_value = neighbor_value

            cf_results_list.append(solution_value)
            evolved_values.append(best_solution_value)

        if current_temp > final_temp:
            current_temp = initial_temp * alpha ** ((ite - fes_count) / neighbor_call_count)
    all_searches_arrays.append(evolved_values)
    all_searches_arrays_best_values.append(min(evolved_values))


def get_neighbors(state, rang):
    """Returns neighbors of the argument state for your solution."""
    nbr = []
    for s in state:
        new_val = random.gauss(s, rang / 10)
        while new_val > rang or new_val < -rang:
            new_val = random.gauss(s, rang / 10)
        nbr.append(new_val)
    return nbr


def ploter(ite, runs, dim, m_name, caller):
    field_cumulative = []
    cumulant = 0
    for j in range(ite):
        for i in range(runs):
            cumulant += all_searches_arrays[i][j]
        field_cumulative.append(cumulant / len(all_searches_arrays))
        cumulant = 0

    xlabtext = "Generations (" + str(ite) + ")"

    cumulative_comp_list.append(field_cumulative)

    plotx = plt
    plotx.plot(field_cumulative)
    plotx.title("Average convergence (" + str(runs) + ") dimension: " +
                str(dim) + "\n" + m_name + " - " + caller)
    plotx.xlabel(xlabtext)
    plotx.yscale("log")
    plotx.ylabel("CF values")
    plotx.show()

    ploty = plt
    for data in all_searches_arrays:
        ploty.plot(data)
    ploty.title("Evolution of CF Value for All Solutions (" + str(runs) +
                ") dimension: " + str(dim) + "\n" + m_name + " - " + caller)
    ploty.xlabel(xlabtext)
    ploty.ylabel("CF values")
    ploty.show()


def runner(method: str, runs, func: int, dim, ran, search, shower, ite=10000):
    """
    Runs the specified algorithm with the given parameters.
    :param method: [RS for random search] [SA for sim. aneal.].
    :type method: int
    :param runs: The number of runs to perform.
    :type runs: int
    :param func: The method to use for the algorithm. [1: dejong] [2: dejong_sec] [3: swafel]
    :type func: str
    :param dim: The dimensionality of the input data.
    :type dim: int
    :param ran: The range of values for the input data.
    :type ran: int
    :param search: The initial most wanted value.
    :type search: int
    :param shower: True for statistics on || False to off
    :type shower: bool
    :param ite: The number of iterations.
    :type ite: int
    """
    cost_function: staticmethod
    if func == 1:
        cost_function = evaluate_de_jong
    elif func == 2:
        cost_function = evaluate_de_jong_sec
    else:
        cost_function = evaluate_schwefel

    testing_method: str
    if method == 'RS':
        testing_method = "Random search"
    elif method == 'SA':
        testing_method = "Simulated annealing"
    else:
        print("testing method not defined, bruh")
        return

    start = time.time()
    if testing_method == "Simulated annealing":
        for i in range(runs):
            simulated_annealinger(cost_function, dim, ran, search, ite)
    elif testing_method == "Random search":
        for i in range(runs):
            random_searcher(cost_function, dim, ran, search, ite)
    else:
        print("testing method not defined, bruh")
        return
    end = time.time() - start

    print(testing_method)
    if shower:
        print(cost_function.__name__ + " in " + str(runs) + " runs | " +
              str(dim) + " dimensions | " + str(ite) + " iterations | " +
              "range -" + str(ran) + " to +" + str(ran) + " | " +
              "search value " + str(search))
        ploter(ite, runs, dim, cost_function.__name__, testing_method)
    else:
        print(min(all_searches_arrays_best_values))

    statictron(testing_method, shower)
    print("(" + str(round(end, 3)) + " sec)\n")


def statictron(fu, shower):
    mini: int
    if fu == "Random search":
        global mini1
        mini1 = min(all_searches_arrays_best_values)
        mini = mini1
    else:
        global mini2
        mini2 = min(all_searches_arrays_best_values)
        mini = mini2
    maxi = max(all_searches_arrays_best_values)
    mean = statistics.mean(all_searches_arrays_best_values)
    medi = statistics.median(all_searches_arrays_best_values)

    if shower:
        print("min: " + str(mini) + " max: " + str(maxi) + " mean: " + str(mean) + " medi: " + str(medi))

    all_searches_arrays.clear()
    all_searches_arrays_best_values.clear()


if __name__ == '__main__':

    runner('RS', 30, 1, 5, 5, 10, True)
    runner('SA', 30, 1, 5, 5, 20, True)

    runner('RS', 30, 1, 10, 5, 60, True)
    runner('SA', 30, 1, 10, 5, 60, True)

    runner('RS', 30, 2, 5, 5, 500, True)
    runner('SA', 30, 2, 5, 5, 150, True)

    runner('RS', 30, 2, 10, 5, 10000, True)
    runner('SA', 30, 2, 10, 5, 600, True)

    runner('RS', 30, 3, 5, 500, 1500, True)
    runner('SA', 30, 3, 5, 500, 1500, True)

    runner('RS', 30, 3, 10, 500, 3000, True)
    runner('SA', 30, 3, 10, 500, 3000, True)

    # plotr = plt
    # plotr.plot(cumulative_comp_list[0], label='RS')
    # plotr.plot(cumulative_comp_list[1], label='SA')
    # plotr.yscale("log")
    # plt.legend()
    # plotr.show()

    if mini1 / mini2 > 0:
        print("sme lepši " + str(round(mini1 / mini2, 2)) + "X")
    else:
        print("sme horši, uch")