"""
Supporting material for the book
Verifying Cyberphysical Systems, by Sayan Mitra

Partial Python program for Checking inductive invariant of
Dijkstra's token ring algorithm for Mutual Exclusion 
Automaton name in the book DijkstrTR

:authors:
    - Chiao Hsieh
    - Sayan Mitra
"""

from z3 import Int, And, Or, Not, Implies, Solver, AtMost, AtLeast
import z3

# Choose a value for the number of processes; try with different values
N = 16
# Choose a value for the maximum value taken by each process
# K has to be greater than N; try different values
K = 32


def bounds(x_list):
    """
    Generate a Z3 Boolean expression that bounds each variable in [0, K)
    :param x_list: list of Z3 variables
    :return: Z3 Boolean expression bounding all variables
    """
    return And(*[And(x >= 0, x < K) for x in x_list])  # TODO


def has_token(x_list, j):
    """
    Generate a Z3 Boolean expression that represents whether P_j holds the token
    :param x_list: list of Z3 variables
    :param j: index of P_j
    :return: Z3 Boolean expression that if true then P_j is holding the token
    """
    token_holder = -1  # Default to no token holder
    for i in range(len(x_list)):
        if x_list[i] == (x_list[(i - 1) % len(x_list)] + 1) % K:
            if token_holder == -1:
                token_holder = i  # First valid token holder found
            else:
                return False  # More than one token holder, invalid state
    
    return z3.BoolVal(True) if token_holder == j else z3.BoolVal(False)  # Return True if P_j is the token holder


  # TODO


def legal_config(x_list):
    """
    Generate a Z3 Boolean expression that represents whether the system is in a legal configuration
    :param x_list: list of Z3 variables
    :return: Z3 Boolean expression that if true then the system is in legal configuration
    """
    # Do not change
    args = [has_token(x_list, i) for i in range(0, N)]
    return And(AtMost(args + [1]), AtLeast(args + [1]))


def transition_relation(old_x_list, new_x_list):
    """
    Generate a Z3 Boolean expression representing transition
    :param old_x_list: variables before transition
    :param new_x_list: variables after transition
    :return: Z3 Boolean expression that when true means the old state can transit to the new state
    """
    constraints = []
    update_conditions = []

    for i in range(N):
        can_update = And(
            has_token(old_x_list, i),  
            new_x_list[i] == (old_x_list[i] + 1) % K  
        )

        constraints.append(Or(
            new_x_list[i] == old_x_list[i],
            can_update  
        ))

        update_conditions.append(can_update)


    return And(*constraints, AtMost(*update_conditions, 1), AtLeast(*update_conditions, 1))

def invariant(x_list):
    """
    Generate a Z3 Boolean expression representing invariant
    :param x_list: list of Z3 variables
    :return: Z3 Boolean expression of invariant
    """
    # Do not change
    return legal_config(x_list)


def prove(conjecture):
    # Do not change
    # Setting up solver
    s = Solver()
    # s.set("sat.cardinality.solver", True)  # Some options to speed up the solver

    s.add(Not(conjecture))  # Check unsat of negation for checking validity
    result = s.check()
    if result == z3.sat:
        print("Given formula is not valid.")
        print("Counter example: \n", s.model())
    elif result == z3.unsat:
        print("Given formula is valid.")
    else:  # result == z3.unknown
        print("Inconclusive. Z3 cannot solve with given options.")


def main():
    # 1. Create a list of z3 Int variables called prestate which is [x[0], x[1], x[2],....,x[15]]
    # use the range() function in Python and the parameter N
    # do not hardcode the list explicitly
    prestate = [Int(f"x[{i}]") for i in range(N)]  # TODO
    # 2. create a list of z3 Int variables called poststate which is [x'[0], x'[1], x'[2],....,x'[N-1]]
    poststate = [Int(f"x'[{i}]") for i in range(N)]  # TODO

    # Do not change
    # Defines init as prestate that is legal
    init = legal_config(prestate)

    # 3. Write the base_case predicate using the Implies() function of z3
    base_case = Implies(init, invariant(prestate))  # TODO

    # 4. Write the induction step predicate using the Implies() function of z3
    # prestate and poststate and transition_relation
    ind_case = Implies(
        And(invariant(prestate), bounds(prestate), transition_relation(prestate, poststate)),
        And(invariant(poststate), bounds(poststate))
        )  # TODO

    # Do not change
    print("## Proving base case:")
    prove(base_case)
    print("## Proving induction case")
    prove(ind_case)


if __name__ == "__main__":
    main()

