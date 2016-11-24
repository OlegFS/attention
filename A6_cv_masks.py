import attention
reload(attention)
from attention import *


use_sim = False

#### load event data
if not locals().has_key('event_data'):
    if use_sim:
        event_data = cPickle.load(open('data/sim_event_data.v2.pickle'))
        N_reps = 10
    else:
        event_data = cPickle.load(open('data/event_data.pickle'))
        N_reps = 50



"""
=================
Setting up for CV
=================
"""

def setup_cv( d, N_reps=N_reps ):

    # values
    cL, cR = d.is_trial_cue_L, d.is_trial_cue_R
    T, M = d.N_events, d.N_units

    # container for masks
    masks = np.zeros((N_reps, M, T), dtype=bool)

    # cycle through repeats
    for rep in progress.dots( range(N_reps) ):

        # separate into training and testing
        mask = np.zeros((M, T), dtype=bool)
        for t in range(T):
            mask[ perm( M )[:M*0.2], t ] = True

        # save mask
        masks[rep, :, :] = mask

        # is this done already
        try:
            d.load_attribute('cv_masks__int')
            print '[already done]'
            return None
        except IOError:
            pass

    # 10reps fix
    if d.can_load_attribute( 'cv_masks_10reps__int' ):
        d.load_attribute('cv_masks_10reps__int')
        masks[ :10, :, : ] = d.cv_masks_10reps__int

    # return
    return masks


########
# batch 
########


for d in progress.numbers( np.random.permutation( event_data ) ):
    try:
        d.load_attribute('cv_masks__int')
    except IOError:
        d.cv_masks__int = setup_cv(d)
        if d.cv_masks__int is not None:
            d.save_attribute('cv_masks__int', overwrite=None)
