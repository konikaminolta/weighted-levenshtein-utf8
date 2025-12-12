import numpy as np

# Infinity constant
DTYPE_MAX = float('inf')

def _to_ords(s):
    """
    Helper to convert a string to a list of Unicode integer code points.
    'a' -> 97, '😊' -> 128522
    """
    if isinstance(s, str):
        return [ord(c) for c in s]
    if isinstance(s, (bytes, bytearray)):
        return [b for b in s]
    return s

def _get_cost_v1(cost_dict, key, default=1.0):
    """Helper to look up 1D costs (insert/delete) safely."""
    if cost_dict is None:
        return default
    return cost_dict.get(key, default)

def _get_cost_v2(cost_dict, k1, k2, default=1.0):
    """Helper to look up 2D costs (substitute/transpose) safely."""
    if cost_dict is None:
        return default
    return cost_dict.get((k1, k2), default)

def _col_delete_range_cost(d, start, end):
    """
    Calculates cost to delete range based on the DP matrix cumulative sums.
    Logical column 0 is NumPy column 1.
    """
    # Map: index i -> i + 1
    return d[end + 1, 1] - d[start, 1]

def _row_insert_range_cost(d, start, end):
    """
    Calculates cost to insert range based on the DP matrix cumulative sums.
    Logical row 0 is NumPy row 1.
    """
    # Map: index i -> i + 1
    return d[1, end + 1] - d[1, start]

def damerau_levenshtein(
    str1,
    str2,
    insert_costs=None,
    delete_costs=None,
    substitute_costs=None,
    transpose_costs=None
):
    """
    Calculates the Damerau-Levenshtein distance supporting full Unicode.
    
    :param insert_costs: dict {char_code: cost, ...} or None (default 1.0)
    :param delete_costs: dict {char_code: cost, ...} or None (default 1.0)
    :param substitute_costs: dict {(char_a, char_b): cost, ...} or None (default 1.0)
    :param transpose_costs: dict {(char_a, char_b): cost, ...} or None (default 1.0)
    """
    s1 = _to_ords(str1)
    s2 = _to_ords(str2)
    len1 = len(s1)
    len2 = len(s2)

    # da: Dictionary to store the last seen index of a character.
    # Replaces the fixed-size array 'da[ALPHABET_SIZE]'.
    da = {}

    # The DP matrix 'd'
    # Size depends on string length, not alphabet size, so this remains a dense array.
    # Logical -1 -> NumPy 0, Logical 0 -> NumPy 1
    d = np.zeros((len1 + 2, len2 + 2), dtype=float)

    # Fill row (-1) and column (-1) with infinity (NumPy index 0)
    d[0, 0] = DTYPE_MAX
    for i in range(len1 + 1):
        d[i + 1, 0] = DTYPE_MAX
    for j in range(len2 + 1):
        d[0, j + 1] = DTYPE_MAX

    # Fill row 0 and column 0 with insertion and deletion costs (NumPy index 1)
    d[1, 1] = 0.0
    
    # Fill deletion costs (Logical col 0)
    for i in range(1, len1 + 1):
        char_i = s1[i - 1]
        cost = _get_cost_v1(delete_costs, char_i)
        d[i + 1, 1] = d[i, 1] + cost

    # Fill insertion costs (Logical row 0)
    for j in range(1, len2 + 1):
        char_j = s2[j - 1]
        cost = _get_cost_v1(insert_costs, char_j)
        d[1, j + 1] = d[1, j] + cost

    # Fill DP array
    for i in range(1, len1 + 1):
        char_i = s1[i - 1]
        db = 0
        
        for j in range(1, len2 + 1):
            char_j = s2[j - 1]
            
            # Retrieve last seen index from dictionary, default 0
            k = da.get(char_j, 0)
            l = db
            
            if char_i == char_j:
                cost = 0
                db = j
            else:
                cost = _get_cost_v2(substitute_costs, char_i, char_j)

            # 1. Substitute/Match
            c_sub = d[i, j] + cost
            
            # 2. Insert
            c_ins = d[i + 1, j] + _get_cost_v1(insert_costs, char_j)
            
            # 3. Delete
            c_del = d[i, j + 1] + _get_cost_v1(delete_costs, char_i)
            
            # 4. Transpose
            # Accessing chars for transpose cost logic needs care with indices
            trans_char_1 = s1[k - 1] if k > 0 else 0
            trans_char_2 = s1[i - 1]
            
            # Cost calculation
            c_trans = (d[k, l] + 
                       _col_delete_range_cost(d, k + 1, i - 1) + 
                       _get_cost_v2(transpose_costs, trans_char_1, trans_char_2) +
                       _row_insert_range_cost(d, l + 1, j - 1))

            d[i + 1, j + 1] = min(c_sub, c_ins, c_del, c_trans)

        da[char_i] = i

    return d[len1 + 1, len2 + 1]

dam_lev = damerau_levenshtein


def optimal_string_alignment(
    str1,
    str2,
    insert_costs=None,
    delete_costs=None,
    substitute_costs=None,
    transpose_costs=None
):
    """
    Calculates the Optimal String Alignment distance supporting full Unicode.
    """
    s1 = _to_ords(str1)
    s2 = _to_ords(str2)
    len1 = len(s1)
    len2 = len(s2)

    d = np.zeros((len1 + 1, len2 + 1), dtype=float)

    d[0, 0] = 0.0
    for i in range(1, len1 + 1):
        char_i = s1[i - 1]
        d[i, 0] = d[i - 1, 0] + _get_cost_v1(delete_costs, char_i)
    
    for j in range(1, len2 + 1):
        char_j = s2[j - 1]
        d[0, j] = d[0, j - 1] + _get_cost_v1(insert_costs, char_j)

    for i in range(1, len1 + 1):
        char_i = s1[i - 1]
        for j in range(1, len2 + 1):
            char_j = s2[j - 1]
            
            if char_i == char_j:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + _get_cost_v1(delete_costs, char_i),
                    d[i, j - 1] + _get_cost_v1(insert_costs, char_j),
                    d[i - 1, j - 1] + _get_cost_v2(substitute_costs, char_i, char_j)
                )

            if i > 1 and j > 1:
                prev_char_i = s1[i - 2]
                prev_char_j = s2[j - 2]
                
                if char_i == prev_char_j and prev_char_i == char_j:
                    d[i, j] = min(
                        d[i, j],
                        d[i - 2, j - 2] + _get_cost_v2(transpose_costs, prev_char_i, char_i)
                    )

    return d[len1, len2]

osa = optimal_string_alignment


def levenshtein(
    str1,
    str2,
    insert_costs=None,
    delete_costs=None,
    substitute_costs=None
):
    """
    Calculates the Levenshtein distance supporting full Unicode.
    """
    s1 = _to_ords(str1)
    s2 = _to_ords(str2)
    len1 = len(s1)
    len2 = len(s2)

    d = np.zeros((len1 + 1, len2 + 1), dtype=float)

    d[0, 0] = 0.0
    for i in range(1, len1 + 1):
        char_i = s1[i - 1]
        d[i, 0] = d[i - 1, 0] + _get_cost_v1(delete_costs, char_i)
        
    for j in range(1, len2 + 1):
        char_j = s2[j - 1]
        d[0, j] = d[0, j - 1] + _get_cost_v1(insert_costs, char_j)

    for i in range(1, len1 + 1):
        char_i = s1[i - 1]
        for j in range(1, len2 + 1):
            char_j = s2[j - 1]
            
            if char_i == char_j:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + _get_cost_v1(delete_costs, char_i),
                    d[i, j - 1] + _get_cost_v1(insert_costs, char_j),
                    d[i - 1, j - 1] + _get_cost_v2(substitute_costs, char_i, char_j)
                )

    return d[len1, len2]

lev = levenshtein