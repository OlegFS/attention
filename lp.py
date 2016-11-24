def profline( statement, functions ):
    from line_profiler import LineProfiler
    try:
        len( functions )
    except TypeError:
        lp = LineProfiler( functions )
    else:
        lp = LineProfiler( *functions )
    lp.run( statement )
    lp.print_stats()


