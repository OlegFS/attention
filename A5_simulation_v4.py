import attention
reload(attention)
from attention import *

repeat_idx = 15
target_filename = 'data/sim_event_data.v4.repeat.%d.pickle' % repeat_idx
true_ranks = [1, 2, 3, 4, 6, 8, 12]

event_data = cPickle.load(open('data/event_data.pickle'))

keys_to_remove = ['trial_end_code', 'prev_trial_end_code', 'event_idxs_within_trial', 'N_events_within_trial', 'trial_idxs_within_block', 'is_trial_catch', 'prev_prev_trial_target_R', 'target_orientation_change', 'prev_prev_trial_catch', 'prev_prev_trial_end_code', 'prev_trial_catch', 'prev_trial_target_R', 'trial_block_idxs', 'trial_idxs']


### SIMULATE

if os.path.exists(target_filename):
    print 'loading...'
    sim_event_data = cPickle.load(open(target_filename))

else:

    # container
    sds = []

    # get cell
    for d_idx, d in enumerate( progress.numbers( event_data ) ):
    
        N, T = d.N_units, d.N_events

        # source model
        d.load_attribute(['CGL', 'CGLM_2'])
        m = Model( d.CGLM_2 )

        # how to determine the right gain
        """
        print 'calculating gain...'

        def get_nc( gain ):
            rank = 6
            W__in = np.random.normal( m.W__in.mean(), m.W__in.std(), size=(rank, N) )
            S__ti = np.random.normal( m.S__ti.mean(), m.S__ti.std(), size=(T, rank) )
            Omega__tn = S__ti.dot(W__in)
            Omega__tn -= Omega__tn.mean()
            Omega__tn *= gain / Omega__tn.std()
            Mu__tn = d.CGL.G__tn * np.exp( Omega__tn + d.S__ts.dot(d.CGL.C__sn) )
            Y__tn = np.random.poisson( Mu__tn )
            sd = d.__copy__()
            sd.spike_counts = Y__tn.T
            return np.median(sd.noise_correlations)
        def get_nc_reps( gain ):
            return np.median([ get_nc(gain) for _ in range(40) ])

        true_nc = np.median(d.noise_correlations)
        from scipy.optimize import fmin
        func = lambda g: ( get_nc_reps(g) - true_nc )**2
        gain = fmin( func, 0.1, xtol=1e-3, ftol=1e-3, maxiter=20 )[0]
        """

        d.load_attribute('sim_Omega_gain')
        gain = d.sim_Omega_gain

        # new cells
        for rank in true_ranks:
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
            sd = d.__copy__()
            sd.spike_counts = Y__tn.T
            sd.Omega_gain = gain
            sd.prefix = 'sim_v4_repeat_%d_source_%d_rank_%d' % (repeat_idx, d_idx, rank)
            # metadata
            sd.sim_repeat_idx = repeat_idx
            sd.sim_source = d_idx
            sd.sim_rank = rank
            # remove keys
            for k in keys_to_remove:
                delattr( sd, k )
            delattr( sd, 'CGL' )
            delattr( sd, 'CGLM_2' )
            # save
            sds.append( sd )

        sim_event_data = EventDataList( sds )

    # save
    cPickle.dump( sim_event_data, open(target_filename, 'w'), 2 )


### COPY OVER PARAMETERS

attrs_to_copy = ['CGL', 'cv_masks__int']
d.load_attribute(attrs_to_copy)
for a in attrs_to_copy:
    for sd in sim_event_data:
        src = event_data[ sd.sim_source ]._attr_filename(a)
        dst = sd._attr_filename(a)
        os.symlink( src, dst )

