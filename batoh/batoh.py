import itertools
import math
import random
import time
import matplotlib.pyplot as plt

supr_pole = []

"""             
DISCLAIMER:     
this project was done by multiple terrestrials:
Tomáš Polívka                                
Lukáš Gazdík                                    (°.° )
Marek Trefil
Dominik Vodička
"""

# item, weight, value
ITEMS = (
    ["hotdog", 25, 20],
    ["knife", 20, 30],
    ["mobile phone", 10, 46],
    ["socks", 15, 5],
    ["book", 30, 28],
    ["tshirt", 30, 11],
    ["flash light", 19, 15],
    ["water bottle", 30, 32],
    ["extra hot chlli", 11, 64],
    ["medic kit", 45, 45],
    ["sony headphones", 12, 48],
    ["Heckler & Koch USP9", 79, 56],
    ["5 €", 1, 5],
    ["watches", 7, 20],
    ["glue", 6, 10],
    ["super glue", 12, 20],
    ["super mega glue", 24, 30],
    ["coffe", 13, 19],
    ["oreo mega pack", 20, 31],
    ["wireless charger", 18, 45]
)


def print_batoh_items(mask):
    b_i = []
    for i in range(len(mask)):
        if mask[i]:
            b_i.append(ITEMS[i])

    print(f"Item\t\t\t\t size  value")
    for i in b_i:
        print(f"{i[0]:<20} {i[1]:<5} {i[2]}")
    print(f"items: {len(b_i)}, size: {get_v_w(b_i, 'W')}, value: {get_v_w(b_i, 'V')}")


def generate_random_items(items):
    return [random.randint(0, 1) == 1 for _ in range(len(items))]


def simulated_annealinger(items, max_weight_limit, ite=10000):
    start = time.time()

    fes_count = ite
    neighbor_call_count = 10
    alpha = 0.99
    initial_temp = 1000
    final_temp = 0.01
    current_temp = initial_temp

    all_solutions = []

    current_items = generate_random_items(items)
    solution_value = cost_function_masking(current_items)
    best_solution_value = 0
    best_neighbour = solution_value
    all_solutions.append(best_solution_value)

    while fes_count > 0:
        for _ in range(neighbor_call_count):
            fes_count -= 1
            neighbor_items = get_neighbors(current_items, max_weight_limit)
            neighbor_value = cost_function_masking(neighbor_items)
            cost_diff = neighbor_value - solution_value
            if cost_diff > 0:
                current_items = neighbor_items
                solution_value = neighbor_value
                if solution_value > best_solution_value:
                    best_solution_value = solution_value
                    best_neighbour = neighbor_items
            else:
                prob = math.exp(cost_diff / current_temp)
                if random.uniform(0, 1) < prob:
                    current_items = neighbor_items
                    solution_value = neighbor_value
            all_solutions.append(best_solution_value)
        if current_temp > final_temp:
            current_temp = initial_temp * alpha ** ((ite-fes_count)/neighbor_call_count)

    ###########################################
    # pll = plt
    # pll.title("anealing")
    # # pll.yscale("log")
    # pll.plot(all_solutions)
    # pll.show()
    ###########################################
    supr_pole.append(all_solutions)
    print(time.time() - start)
    return best_neighbour


def get_neighbors(state, max_limit):
    nbr = []
    chance_to_change = 1 / len(ITEMS)
    t_or_f = True
    while t_or_f:
        nbr = []
        for dim_val in state:
            if random.uniform(0, 1) < chance_to_change:
                nbr.append(not dim_val)
            else:
                nbr.append(dim_val)
        t_or_f = not get_weight(max_limit, nbr)
    return nbr


def brute_forcer(b_i, b_m_l):
    start = time.time()
    masks = list(itertools.product([True, False], repeat=len(b_i)))
    batohy = []
    kumulativa_bruteforce = []
    x = 0
    for m in masks:
        if get_weight(b_m_l, m):
            v = cost_function_masking(m)
            batohy.append(v)
            if v > x:
                x = v
        else:
            batohy.append(0)
        kumulativa_bruteforce.append(x)
    ###########################################
    # pll = plt
    # pll.title("brute")
    # # pll.yscale("log")
    # pll.plot(batohy)
    # pll.show()
    ###########################################
    supr_pole.append(kumulativa_bruteforce)
    print(time.time()-start)
    return masks[batohy.index(max(batohy))]


def get_v_w(bat, st):
    x = 0
    if st == 'W':
        for i in bat:
            x += i[1]
    elif st == 'V':
        for i in bat:
            x += i[2]
    else:
        return 0
    return x


def cost_function_masking(mask):
    """
    Takes mask to return its value
    """
    v = 0
    for m in range(len(mask)):
        if mask[m]:
            v += ITEMS[m][2]
    return v


def get_weight(max_w, mask):
    """
    True when weight not above limit
    """
    w = 0
    for m in range(len(mask)):
        if mask[m]:
            w += ITEMS[m][1]
    return w <= max_w


if __name__ == '__main__':
    BATOH_LIMIT = 100 if len(ITEMS) < 10 else 150 if len(ITEMS) < 15 else 200
    print(len(ITEMS))
    print(BATOH_LIMIT)

    pll = plt
    for _ in range(30):
        simulated_annealinger(ITEMS, BATOH_LIMIT)
    for s in supr_pole:
        pll.plot(s)
    pll.title("anealing in " + str(30) + " runs")
    pll.show()

    brute_forcer(ITEMS, BATOH_LIMIT)

    plo = plt
    plo.title("bruteforce anealing comparison")
    # plo.yscale("log")
    lab1 = "brute max: " + str(max(supr_pole[len(supr_pole)-1]))
    plo.plot(supr_pole[len(supr_pole)-1], label=lab1)
    lab2 = "anealing max: " + str(max(supr_pole[1]))
    plo.plot(supr_pole[1], label=lab2)
    plo.xlim(0, 12000)
    plt.legend()
    plo.show()