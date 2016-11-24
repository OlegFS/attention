import attention
reload(attention)
from attention import *

repeat_idx = 0
target_filename = 'data/sim_event_data.v4.repeat.%d.pickle' % repeat_idx
true_ranks = [1, 2, 3, 4, 6, 8, 12]

event_data = cPickle.load(open('data/event_data.pickle'))

keys_to_remove = ['trial_end_code', 'prev_trial_end_code', 'event_idxs_within_trial', 'N_events_within_trial', 'trial_idxs_within_block', 'is_trial_catch', 'prev_prev_trial_target_R', 'target_orientation_change', 'prev_prev_trial_catch', 'prev_prev_trial_end_code', 'prev_trial_catch', 'prev_trial_target_R', 'trial_block_idxs', 'trial_idxs']


### SIMULATE

# get cell
for d_idx in progress.dots( np.random.permutation(range(36)) ):

    d = event_data[d_idx]
    print d_idx, d.__repr__()

    if d.can_load_attribute('sim_Omega_gain'):
        continue

    N, T = d.N_units, d.N_events

    # source model
    d.load_attribute(['CGL', 'CGLM_2'])
    m = Model( d.CGLM_2 )

    # how to determine the right gain
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

    d.sim_Omega_gain = gain
    d.save_attribute('sim_Omega_gain', overwrite=None)
