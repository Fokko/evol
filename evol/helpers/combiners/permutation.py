from itertools import islice, tee
from random import choice
from typing import Any, Sequence, Tuple

from ._utils import select_node, construct_neighbors, multiple_offspring, identify_cycles, cycle_parity
from .._utils import select_partition


def order_one_crossover(parents: Sequence[Tuple]) -> Tuple:
    """Combine two chromosomes using order-1 crossover.

    http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/Order1CrossoverOperator.aspx

    :param parents: Sequence of two parents.
    :return: Child chromosome.
    """
    assert len(parents) == 2, 'The combiner order_one_crossover accepts two parents. Got {n}'.format(n=len(parents))
    start, end = select_partition(len(parents[0]))
    selected_partition = parents[0][start:end+1]
    remaining_elements = filter(lambda element: element not in selected_partition, parents[1])
    return tuple(islice(remaining_elements, 0, start)) + selected_partition + tuple(remaining_elements)


def edge_recombination(parents: Tuple) -> Tuple:
    """Combine multiple chromosomes using edge recombination.

    http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/EdgeRecombinationCrossoverOperator.aspx

    :param parents: Chromosomes to combine.
    :return: Child chromosome.
    """
    return tuple(select_node(
        start_node=choice([chromosome[0] for chromosome in parents]),
        neighbors=construct_neighbors(*parents)
    ))


@multiple_offspring
def cycle_crossover(parents: Sequence[Tuple]) -> Tuple[Tuple[Any, ...], ...]:
    """Combine two chromosomes using cycle crossover.

    http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/CycleCrossoverOperator.aspx

    :param parents: Sequence of two parents.
    :return: Tuple of two child chromosomes.
    """
    assert len(parents) == 2, 'The combiner cycle_crossover accepts two parents. Got {n}'.format(n=len(parents))
    cycles = identify_cycles(parents[0], parents[1])
    parity = cycle_parity(cycles=cycles)
    it_a, it_b = tee((b, a) if parity[i] else (a, b) for i, (a, b) in enumerate(zip(parents[0], parents[1])))
    return tuple(x[0] for x in it_a), tuple(y[1] for y in it_b)
