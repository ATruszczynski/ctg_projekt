class AlgoTuple:
    pass

class GA_Tuple (AlgoTuple):
    def __init__(self, repetitions:int, pop_count: int, iterations: int, mprob: float, cprob: float, selected: int, verbal: int = 0,
                 patience: int = 5, fix_prob: float = 0.1, mutate_ver_prob: float = 0.01,
                 random_init: bool = False, pool_count: int = 12):
        self.repetitions = repetitions
        self.pop_count = pop_count
        self.iterations = iterations
        self.mprob = mprob
        self.cprob = cprob
        self.selected = selected
        self.verbal = verbal
        self.pool_count = pool_count
        self.patience = patience
        self.fix_prob = fix_prob
        self.mutate_ver_prob = mutate_ver_prob
        self.random_init = random_init
