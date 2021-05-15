class AlgoTuple:
    def __init__(self, repetitions: int):
        self.repetitions = repetitions
    pass


class GA_Tuple (AlgoTuple):
    def __init__(self, repetitions:int, pop_count: int, iterations: int, mprob: float, cprob: float, selected: int, verbal: int = 0,
                 patience: int = 5, fix_prob: float = 0.1,
                 random_init: bool = False, pool_count: int = 12):
        super().__init__(repetitions)
        self.pop_count = pop_count
        self.iterations = iterations
        self.mprob = mprob
        self.cprob = cprob
        self.selected = selected
        self.verbal = verbal
        self.pool_count = pool_count
        self.patience = patience
        self.fix_prob = fix_prob
        self.random_init = random_init


class Pure_Greed_Tuple(AlgoTuple):
    def __init__(self, repetitions: int):
        super().__init__(repetitions)


class Greed_Tuple(AlgoTuple):
    def __init__(self, repetitions: int):
        super().__init__(repetitions)


class DSatur_Tuple(AlgoTuple):
    def __init__(self, repetitions: int):
        super().__init__(repetitions)

class RVC_Tuple(AlgoTuple):
    def __init__(self, repetitions: int):
        super().__init__(repetitions)




