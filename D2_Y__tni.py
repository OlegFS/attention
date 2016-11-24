import attention
reload(attention)
from attention import *


"""
=========
Load data
=========
"""

event_data = cPickle.load(open('data/event_data.pickle'))
raw_data = cPickle.load(open('data/raw_data.pickle'))
N_data = len(event_data)


N_bins = 60


def compute_Y__tni( d_idx ):
    # get data
    d = event_data[ d_idx ]
    r = raw_data[d_idx]
    for k in keys_to_load:
        d.load_attribute(k)
    # is it done
    if 'Y__tni' in d.available_attributes:
        print '[already done]'
        return None
    # collect Y
    Y__tni = np.zeros( (d.N_events, d.N_units, N_bins), dtype=float )
    for t in progress.dots( range( d.N_events ), 'collecting Y' ):
        trial_idx = d.trial_idxs[ t ]
        event_in_trial_idx = d.event_idxs_within_trial[ t ]
        t_start = r[ trial_idx ].t_msec_relative_stimulus_on[ 
                event_in_trial_idx ] - 100
        t_end = t_start + N_bins * 10
        bin_edges = np.arange(t_start, t_end+1, 10)
        Y__tni[t, :, :] = A([ 
            np.histogram( 
                r[trial_idx].t_spikes[unit_idx], bins=bin_edges )[0] 
            for unit_idx in range(d.N_units) ])
        # is it done
        if 'Y__tni' in d.available_attributes:
            print '[already done]'
            return None
    # save
    d.Y__tni = Y__tni.astype(np.int8)
    d.save_attribute('Y__tni')



# job list
jobs = []
for d_idx in np.random.permutation( range( N_data ) ):
    for f in [compute_Y__tni]:
        jobs.append((f, d_idx))


if __name__ == '__main__':
    # run
    for j in progress.numbers( jobs ):
        func, d = j
        func(d)
    pass
