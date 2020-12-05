from itertools import chain, combinations

def get_powerset(set_elements):
    # Set + empty set + subsets of given set
    list_elements = list(set_elements)
    return list(chain.from_iterable(combinations(list_elements, e) 
        for e in range(1, len(list_elements)+1))) # start at 1 to ignore empty set

def disease_p_a():
    set1 = {"P"}
    set2 = {"A"}
    return set1.union(set2)

if __name__ == "__main__":
    set_elements  = disease_p_a()
    list_powerset = get_powerset(set_elements)
    print(list_powerset)