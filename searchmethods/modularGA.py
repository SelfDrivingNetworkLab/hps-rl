
import numpy, random

class Individual:
    def __init__(self,genome, llimits =[], ulimits=[], type=[], LEN = 1,fitness_func = None):
        if genome is None:
            self.genome = numpy.zeros(LEN,dtype=float)
            for gene in range(LEN):
                    if type[gene] == "integer":
                        self.genome[gene] = numpy.random.randint(llimits[gene], ulimits[gene])
                    else:
                        self.genome[gene] = numpy.random.uniform(llimits[gene], ulimits[gene])
        else:
            self.genome = genome
        self.fitness = fitness_func(self.genome)

    def __str__(self):
        return "".join(str(int(i)) for i in self.genome)


def crossover(a, b, fitness):
    g, h = a.genome.copy(), b.genome.copy()
    for pt in range(len(g)):
        if numpy.random.random() < 0.5:
            g[pt], h[pt] = h[pt], g[pt]
    return (Individual(genome=g,fitness_func=fitness), Individual(genome=h,fitness_func=fitness))

def mutate(a, mut_prob,fitness):
    g = a.genome.copy()
    for pt in range(len(g)):
        if numpy.random.random() < mut_prob:
            g[pt] = not g[pt]
    return Individual(g,fitness_func=fitness)


def stats(pop, gen,threshold):
    best = max(pop, key=lambda x: x.fitness)
    print("{0} {1:.2f} {2} {3}".format(gen, numpy.mean([i.fitness for i in pop]), best.fitness, str(best)))
    return (best.fitness >= threshold)


def roulette(items, n):
    total = float(sum(w.fitness for w in items))
    i = 0
    w, v = items[0].fitness, items[0]
    while n:
        x = total * (1 - numpy.random.random() ** (1.0 / n))
        total -= x
        while x > w:
            x -= w
            i += 1
            w, v = items[i].fitness, items[i]
        w -= x
        yield v
        n -= 1


def tournament(items, n, tsize=5):
    for i in range(n):
        candidates = random.sample(items, tsize)
        yield max(candidates, key=lambda x: x.fitness)

def step(pop,cross_prob,mut_prob,fitness):
    newpop = []
    parents = roulette(pop, len(pop) + 1)  # one extra for final xover
    while len(newpop) < len(pop):
        if numpy.random.random() < cross_prob:
            newpop.extend(map(mutate, crossover(next(parents), next(parents),fitness=fitness),[mut_prob,mut_prob],[fitness,fitness]))
        else:
            newpop.append(mutate(next(parents),mut_prob=mut_prob,fitness=fitness))
    return newpop


def run(llimit, ulimit, type, GENERATIONS, CROSSOVER_PROB, POPSIZE, LEN, MUTATION_PROB,FITNESS,THRESHOLD):
    numpy.random.seed(100)
    pop = [Individual(None,llimit,ulimit,type,LEN,FITNESS) for i in range(POPSIZE)]
    print(pop)
    stats(pop, 0, THRESHOLD)
    for gen in range(1, GENERATIONS):
        pop = step(pop,CROSSOVER_PROB,MUTATION_PROB,FITNESS)
        if stats(pop, gen, THRESHOLD):
            print("Success")

llimit = [0.5,1e-6,1e-6,0]
ulimit = [1.5,0.1,0.1,3]
type = ['real','real','real','integer']
LEN = 4
FITNESS, SUCCESS_THRESHOLD = (numpy.sum, LEN)
run(llimit,ulimit,type,100,1,100,4,0.9,FITNESS,10)