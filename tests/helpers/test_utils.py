from pytest import mark, raises

from evol import Population
from evol.helpers.utils import select_arguments, fitness_weights, surviving_population


class TestFitnessWeights:

    def test_weights(self, simple_chromosomes, simple_evaluation_function):
        for maximize in (False, True):
            pop = Population(chromosomes=simple_chromosomes,
                             eval_function=simple_evaluation_function, maximize=maximize)
            with raises(RuntimeError):
                _ = fitness_weights(pop.individuals, maximize=pop.maximize)
            pop.evaluate()
            weights = fitness_weights(pop.individuals, maximize=pop.maximize)
            assert max(weights) == 1
            assert min(weights) == 0
            if maximize:
                assert weights[0] == 0
            else:
                assert weights[0] == 1


class TestSelectArguments:

    @mark.parametrize('args,kwargs,result', [((1, ), {}, 1), ((1, 2), {'x': 1}, 3), ((4, 5), {'z': 8}, 9)])
    def test_no_kwargs(self, args, kwargs, result):
        @select_arguments
        def fct(*args):
            return sum(args)
        assert fct(*args, **kwargs) == result

    @mark.parametrize('args,kwargs,result', [((1, ), {}, 1), ((1, 2), {'x': 1}, 3), ((4, 5), {'z': 8}, 17)])
    def test_with_kwargs(self, args, kwargs, result):
        @select_arguments
        def fct(*args, z=0):
            return sum(args)+z
        assert fct(*args, **kwargs) == result

    @mark.parametrize('args,kwargs,result', [((1,), {'b': 3}, 4), ((1, 2), {'x': 1}, 4), ((4, 5), {'z': 8}, 17)])
    def test_all_kwargs(self, args, kwargs, result):
        @select_arguments
        def fct(a, b=0, **kwargs):
            return a + b + sum(kwargs.values())
        assert fct(*args, **kwargs) == result


class TestSurvivingPopulation:

    def test_base(self):
        assert surviving_population(100, fraction=0.5, n=None, n_islands=1) == 50
        assert surviving_population(100, fraction=0.33, n=None, n_islands=1) == 33
        assert surviving_population(100, fraction=0.336, n=None, n_islands=1) == 34
        assert surviving_population(100, fraction=None, n=10, n_islands=1) == 10

    def test_min(self):
        assert surviving_population(100, fraction=0.5, n=10, n_islands=1) == 10
        assert surviving_population(100, fraction=0.1, n=50, n_islands=1) == 10

    def test_raises(self):
        with raises(ValueError):
            surviving_population(100, fraction=None, n=None, n_islands=1)
        with raises(RuntimeError):
            surviving_population(100, fraction=0, n=None, n_islands=1)
        with raises(RuntimeError):
            surviving_population(10, fraction=0.01, n=None, n_islands=1)
        with raises(RuntimeError):
            surviving_population(100, fraction=0.1, n=None, n_islands=20)
        with raises(ValueError):
            surviving_population(100, fraction=None, n=120, n_islands=1)
        with raises(ValueError):
            surviving_population(100, fraction=1.1, n=None, n_islands=1)

    def test_islands(self):
        assert surviving_population(100, fraction=0.5, n=None, n_islands=2) == 25
        assert surviving_population(100, fraction=0.3, n=None, n_islands=3) == 10
        assert surviving_population(110, fraction=None, n=100, n_islands=3) == 33
        assert surviving_population(100, fraction=None, n=10, n_islands=5) == 2
