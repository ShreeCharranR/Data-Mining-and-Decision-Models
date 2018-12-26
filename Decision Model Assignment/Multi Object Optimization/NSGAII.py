from __future__ import division

import inspyred
import random
import matplotlib.pyplot as plt
import math


def my_generator(random, args):
    return [random.uniform(1e-32, 3.14) for _ in range(2)]


@inspyred.ec.evaluators.evaluator
def my_evaluator(candidate, args):
    x1 = candidate[0]
    x2 = candidate[1]
    c1 = x1 ** 2 + x2 ** 2 - 1 - 0.1 * math.cos(16*math.atan(x1/x2))
    c2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2
    
    if c1 < 0:
        x1 += 3.14 + abs(c1)
        x2 += 3.14 + abs(c1)
    if c2 > 0.5:
        x1 += 3.14 + (c2 - 0.5)
        x2 += 3.14 + (c2 - 0.5)
    return inspyred.ec.emo.Pareto([x1, x2])


def main():
    randseed = 12345
    prng = random.Random(randseed)
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.selector = inspyred.ec.selectors.truncation_selection
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    ea.observer = [inspyred.ec.observers.archive_observer]
    final_pop = ea.evolve(generator=my_generator,
                          evaluator=my_evaluator,
                          pop_size=100,
                          maximize=False,
                          num_selected=10,
                          bounder=inspyred.ec.Bounder(1e-32, 3.14),
                          max_evaluations=5000)
    """num_selected=10,
       tournament_size=5,"""
       
     
    

    display = True
    if display:
        final_arc = ea.archive
        print('Best Solutions: \n')
        for f in final_arc:
            print(f)
        x = []
        y = []
        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])
        plt.scatter(x, y, color='b')
        plt.xlabel("Function 1")
        plt.ylabel("Function 2")
        # Plot and save in pdf format
        # plt.savefig('pf.pdf',format='pdf')
        plt.show()

main()