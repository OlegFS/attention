import attention
reload(attention)
from attention import *

target_filename = 'data/sim_event_data.v3.pickle'
true_ranks = [1, 2, 3, 4, 6, 8, 12, 16, 32]
N_samples = 50

event_data = cPickle.load(open('data/event_data.pickle'))


### SIMULATE


# get cell
d = event_data[13].__copy__()
N, T = d.N_units, d.N_events

if os.path.exists(target_filename):
    print 'loading...'
    sim_event_data = cPickle.load(open(target_filename))

else:
    print 'calculating...'
    # source model
    d.load_attribute(['CGL', 'CGLM_2'])
    m = Model( d.CGLM_2 )

    # parameters
    gain = 0.137

    # new cells
    dsims = []
    for rank in progress.numbers( true_ranks ):
        for sample_idx in progress.dots( range(N_samples), 'rank = %d' % rank ):
            # sample W
            W__in = np.random.normal( m.W__in.mean(), m.W__in.std(), size=(rank, N) )
            # sample S
            S__ti = np.random.normal( m.S__ti.mean(), m.S__ti.std(), size=(T, rank) )
            # create Omega
            Omega__tn = S__ti.dot(W__in)
            # reweight
            Omega__tn -= Omega__tn.mean()
            # multiply by gain
            Omega__tn *= gain / Omega__tn.std()
            # mean rate
            Mu__tn = d.CGL.G__tn * np.exp( Omega__tn + d.S__ts.dot(d.CGL.C__sn) )
            # sample
            Y__tn = np.random.poisson( Mu__tn )
            # replace
            this_d = d.__copy__()
            this_d.spike_counts = Y__tn.T
            this_d.prefix = 'sim_v3_rank_%d_sample_%02d' % (rank, sample_idx)
            # remove keys
            to_remove = ['trial_end_code', 'prev_trial_end_code', 'event_idxs_within_trial', 'N_events_within_trial', 'trial_idxs_within_block', 'is_trial_catch', 'prev_prev_trial_target_R', 'target_orientation_change', 'prev_prev_trial_catch', 'prev_prev_trial_end_code', 'prev_trial_catch', 'prev_trial_target_R', 'trial_block_idxs', 'trial_idxs']
            for k in to_remove:
                delattr( this_d, k )
            # save
            dsims.append( this_d )

    sim_event_data = EventDataList( dsims )

    # save
    cPickle.dump( sim_event_data, open(target_filename, 'w'), 2 )


### COPY OVER PARAMETERS

attrs_to_copy = ['CGL', 'cv_masks__int']
d.load_attribute(attrs_to_copy)
sd0 = sim_event_data[0]
for a in attrs_to_copy:
    setattr( sd0, a, getattr(d, a) )
    sd0.save_attribute( a, overwrite=None )
    src = sd0._attr_filename(a)
    for sd in sim_event_data[1:]:
        dst = sd._attr_filename(a)
        os.symlink( src, dst )


"""
# copy over tau
if not locals().has_key('s1'):
    s1 = cPickle.load(open('data/sim_event_data.v1.pickle'))
for i in range(1, 11):
    attr = 'tau_CGLM_%d' % i
    s1.load_attribute(attr)
    for sd in s1:
        a = getattr(sd, attr)
        try:
            del a['LLs']
        except KeyError:
            pass

s1_prefixes = [sd.prefix for sd in s1]

for sd in progress.dots( sim_event_data, 'copying tau' ):
    this_s1_prefix = sd.prefix.split('_sample')[0]
    this_s1_idx = s1_prefixes.index(this_s1_prefix)
    for i in range(1, 11):
        attr = 'tau_CGLM_%d' % i
        setattr( sd, attr, getattr( s1[this_s1_idx], attr ) )
        sd.save_attribute( attr, overwrite=None )
"""
