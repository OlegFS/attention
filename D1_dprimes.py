# modules
import attention
reload(attention)
from attention import *
from behaviour_glms import *

from scipy.misc import factorial
from scipy.signal import convolve2d

# core data
event_data = cPickle.load(open('data/event_data.pickle'))

if not locals().has_key('raw_data'):
    raw_data = cPickle.load(open('data/raw_data.pickle'))


dprimes_attended = []
dprimes_unattended = []

for data_idx in progress.dots( range(36) ):
    
    this_d = event_data[ data_idx ]
    this_d.dprimes = Bunch({'L':Bunch(), 'R':Bunch(), 'all':Bunch()})

    if this_d.can_load_attribute('dprimes'):
        continue
    
    for side in ['L', 'R']:        

        #data_idx = 0
        #side = 'L'
        
        # attended
        # ---------

        # get data
        rd = raw_data[data_idx]
        d = event_data[data_idx]
        targets = rd.all_event_data.filter_events( rd.all_event_data.event_types == 'target' )

        if side == 'L':
            d = d.filter_rf_L.filter_cue_L
            targets = targets.filter_rf_L.filter_cue_L
            targets = targets.filter_events( targets.is_trial_target_L )
        else:
            d = d.filter_rf_R.filter_cue_R
            targets = targets.filter_rf_R.filter_cue_R
            targets = targets.filter_events( targets.is_trial_target_R )

        # we have to take saccade time into account
        dt = A([ (rd[trial_idx].t_msec_relative_saccade_initiated - rd[trial_idx].t_msec_relative_target_on) 
                for trial_idx in targets.trial_idxs ])
        targets = targets.filter_events( ~(dt < 200) )
        
        # only want middle targets
        targets = targets.filter_events( ( targets.target_orientation_change > 8 ) & ( targets.target_orientation_change < 20 ) )        
        #targets = targets.filter_events( targets.target_orientation_change > 8 )

        # calculate dprime
        standard_mean = standard_mean_attended = d.spike_counts.mean(axis=1)
        standard_var = d.spike_counts.var(axis=1)
        target_mean = targets.spike_counts.mean(axis=1)
        target_var = targets.spike_counts.var(axis=1)
        dprime_attended = (target_mean - standard_mean) / np.sqrt( 0.5*(target_var + standard_var) )
        
        # unattended
        # -----------

        # get data
        rd = raw_data[data_idx]
        d = event_data[data_idx]
        targets = rd.all_event_data.filter_events( rd.all_event_data.event_types == 'target' )

        if side == 'L':
            d = d.filter_rf_L.filter_cue_R
            targets = targets.filter_rf_L.filter_cue_R
            targets = targets.filter_events( targets.is_trial_target_L )
        else:
            d = d.filter_rf_R.filter_cue_L
            targets = targets.filter_rf_R.filter_cue_L
            targets = targets.filter_events( targets.is_trial_target_R )

        # we have to take saccade time into account
        dt = A([ (rd[trial_idx].t_msec_relative_saccade_initiated - rd[trial_idx].t_msec_relative_target_on) 
                for trial_idx in targets.trial_idxs ])
        targets = targets.filter_events( ~(dt < 200) )

        # calculate dprime
        standard_mean = standard_mean_unattended = d.spike_counts.mean(axis=1)
        standard_var = d.spike_counts.var(axis=1)
        target_mean = targets.spike_counts.mean(axis=1)
        target_var = targets.spike_counts.var(axis=1)
        dprime_unattended = (target_mean - standard_mean) / np.sqrt( 0.5*(target_var + standard_var) )

        # save
        this_d.dprimes[side].attended = dprime_attended
        this_d.dprimes[side].unattended = dprime_unattended
        
    # together
    this_d.dprimes.all.attended = np.zeros( this_d.N_units )
    this_d.dprimes.all.unattended = np.zeros( this_d.N_units )
    this_d.dprimes.all.attended[ this_d.is_unit_rf_L ] = this_d.dprimes.L.attended
    this_d.dprimes.all.attended[ this_d.is_unit_rf_R ] = this_d.dprimes.R.attended
    this_d.dprimes.all.unattended[ this_d.is_unit_rf_L ] = this_d.dprimes.L.unattended
    this_d.dprimes.all.unattended[ this_d.is_unit_rf_R ] = this_d.dprimes.R.unattended
    
    this_d.save_attribute('dprimes', overwrite=None)
