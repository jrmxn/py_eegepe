from numpy import sort
from sympy import isprime
from sympy import factorint
from sympy import primefactors


def num_subplots(n):
    """
    p, n = num_subplots(n)

    Purpose
    Calculate how many rows and columns of sub-plots are needed to
    neatly display n subplots.

    Inputs
    n - the desired number of subplots.

    Outputs
    p - a vector length 2 defining the number of rows and number of
        columns required to show n plots.
    [ n - the current number of subplots. This output is used only by
          this function for a recursive call.]



    Example: neatly lay out 13 sub-plots
    >> p=numSubplots(13)
    p =
        3   5
    for i=1:13; subplot(p(1),p(2),i), pcolor(rand(10)), end


    Rob Campbell - January 2010
    James McIntosh - December 2018 (conversion to python from Matlab)
    """

    while isprime(n) and n > 4:
        n = n + 1

    p_dict = factorint(n)
    p_list = [[k for r in range(p_dict[k])] for k in p_dict.keys()]
    p = [item for sublist in p_list for item in sublist]
    # p = primefactors(n)

    if len(p) == 1:
        p.insert(0, 1)
        return p, n

    while len(p) > 2:
        if len(p) >= 4:
            p[0] = p[0] * p[-2]
            p[1] = p[1] * p[-1]
            p.pop(-1)
            p.pop(-1)
        else:
            p[0] = p[0] * p[1]
            p.pop(1)
        p = list(sort(p))

    # Reformat if the column / row ratio is too large: we want a roughly square design
    while p[1] / p[0] > 2.5:
        N = n + 1
        p, n = num_subplots(N) # Recursive!

    return p, n
