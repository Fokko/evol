from inspect import signature
from typing import Callable, Generator, Optional, Sequence, Tuple

from evol import Individual


def fitness_weights(individuals: Sequence[Individual], maximize: bool) -> Tuple[float, ...]:
    """Compute weights for individuals according to their fitness.

    The individual with the worst fitness will have weight 0,
    while the individual with the best fitness will have weight 1.

    :param individuals: Sequence of individuals. These must be evaluated.
    :param maximize: If True, the highest fitness is considered best.
    """
    try:
        min_fitness = min(individual.fitness for individual in individuals)
        max_fitness = max(individual.fitness for individual in individuals)
    except TypeError:
        raise RuntimeError('Individual weights can not be computed if the individuals are not evaluated.')
    if min_fitness == max_fitness:
        return (1.,) * len(individuals)
    elif maximize:
        return tuple((individual.fitness - min_fitness)/(max_fitness-min_fitness) for individual in individuals)
    else:
        return tuple(1-(individual.fitness - min_fitness) / (max_fitness - min_fitness) for individual in individuals)


def offspring_generator(parents: Sequence[Individual],
                        parent_picker: Callable[..., Sequence[Individual]],
                        combiner: Callable,
                        **kwargs) -> Generator[Individual, None, None]:
    """Generator for offspring.

    This helps create the right number of offspring,
    especially in the case of of multiple offspring.
    Offspring are created having the same island_id
    as their first parent.

    :param parents: List of parents.
    :param parent_picker: Function that selects parents.
        Must accept all kwargs passed (i.e. must be decorated by select_arguments).
    :param combiner: Function that combines chromosomes.
        Must accept all kwargs passed (i.e. must be decorated by select_arguments).
    :param kwargs: Arguments
    :returns: Children
    """
    while True:
        # Obtain parent chromosomes
        parents = parent_picker(parents, **kwargs)
        if isinstance(parents, Individual):
            chromosomes = (parents.chromosome,)
        else:
            chromosomes = tuple(individual.chromosome for individual in parents)
        # Create children
        if getattr(combiner, 'multiple_offspring', False):
            for child in combiner(*chromosomes, **kwargs):
                yield Individual(chromosome=child, island_id=parents[0].island_id)
        else:
            yield Individual(chromosome=combiner(*chromosomes, **kwargs), island_id=parents[0].island_id)


def select_arguments(func: Callable) -> Callable:
    """Decorate a function such that it accepts any keyworded arguments.

    The resulting function accepts any arguments, but only arguments that
    the original function accepts are passed. This allows keyworded 
    arguments to be passed to multiple (decorated) functions, even if they
    do not (all) accept these arguments.

    :param func: Function to decorate.
    :return: Callable
    """
    def result(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            return func(*args, **{k: v for k, v in kwargs.items() if k in signature(func).parameters})
    return result


def surviving_population(current_population_size: int, fraction: Optional[float],
                         n: Optional[int], n_islands: int=1) -> int:
    """Compute the size of the surviving population.

    If both fraction and n are specified,
    the minimum resulting population size is taken.

    :param current_population_size: Size of the current population.
    :param fraction: Fraction of the original population that survives.
        Defaults to None.
    :param n: Number of individuals of the population that survive.
        Defaults to None.
    :param n_islands: Number of islands currently in the population.
    :return: Resulting size of each island.
    """
    if n is None:
        if fraction is None:
            raise ValueError('everyone survives! must provide either "fraction" and/or "n".')
        resulting_size = fraction * current_population_size
    elif fraction is None:
        resulting_size = n
    else:
        resulting_size = min(round(fraction * current_population_size), n)
    resulting_size = round(resulting_size / n_islands)
    if resulting_size == 0:
        raise RuntimeError('No one survives!')
    if resulting_size > current_population_size:
        raise ValueError('Everyone survives! Must provide "fraction" < 1 and/or "n" < population size.')
    return resulting_size
