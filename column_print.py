import os
from numpy import ceil

class ColumnError(ValueError):
    pass

def columnise(x, n_cols=None, n_rows=20, space=4, max_n_cols=None):
    if n_cols is None:
        return best_columnise(x, space=space, n_rows=n_rows)
    # how big is the screen
    if max_n_cols is None:
         max_n_cols = int(os.popen('echo $COLUMNS').readlines()[0][:-1])
    # the list of strings
    s = []
    # divvy up into columns
    words_per_col = int(ceil(len(x) / float(n_cols)))
    cols = [x[i*words_per_col:(i+1)*words_per_col] for i in range(n_cols)]
    cols = [c for c in cols if len(c) > 0]
    # how many chars in each column
    col_width = [(max(len(a) for a in c) + space) for c in cols]
    # run through the rows
    n_rows = len(cols[0])
    for r in range(n_rows):
        row = [c[r] for c in cols if len(c) >= (r+1)]
        s.append("".join(word.ljust(col_width[c]) 
            for c, word in enumerate(row)))
    # are any rows too wide?
    if max(len(a) for a in s) > max_n_cols:
        raise ColumnError('n_cols is too big')
    # concatenate
    return reduce(lambda a1, a2: a1 + '\n' + a2, s)

def best_columnise(x, space=4, n_rows=20):
    max_n_cols = int(os.popen('echo $COLUMNS').readlines()[0][:-1])
    # how many rows
    current_n_rows = len(x)
    # want at most n_rows
    n_cols = int(ceil(len(x) / float(n_rows)))
    # work from here down
    s = None
    while s is None:
        try:
            s = columnise(x, n_cols, space=space, max_n_cols=max_n_cols)
        except ColumnError:
            n_cols -= 1
    return s
