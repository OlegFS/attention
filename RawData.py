# %load attention
from helpers import *
from attention import Trial
import models
reload(models)

class RawDataSim( AutoCR ):


    """ Class for loading and processing data from Cohen / Maunsell """
    
    def __init__( self, l,N,TT,ti,t, keep_raw=False, copydict=None ):
        """ Import data from source file. """ 
        self.prefix = 'bla'

        self.l = 1000
        # is this a copy?
        if copydict is not None:
            self.__dict__ = copydict.copy()
            self.trial_block_idx[i]  
            return
        # Generate mock data
        #ti = trialinfo = loadmat(filenames['info'])['trialinfo']
        #ti.instruct = np.zeros([1,self.l])
        #ti.valid = np.ones([1,self.l])
        #ti.catchtrial = np.zeros([1,self.l])
        #ti.attendloc = np.ones([1,self.l])
        #ti.correctloc = np.random.random_integers(0,1,[1,self.l])
        #ti.ochange = np.abs(np.random.normal(0,1,[1,self.l]))
        #ti.eotcode  = zeros([1,self.l])
        r = rates ={'suarrayid':1,'sulist': np.random.poisson(1,[1,self.l])}
        times = {'suarrayid':1,'sulist': np.linspace(1,l,l),'trialtimes':[TT(),TT(),TT()],'spike_counts':np.random.poisson(1,[1,self.l]) }


        """ Pre-determined trial parameters """
        
        # instruction trials: single stimulus, start of each block
        # (should ignore)
        self.is_trial_instruction = ti.instruct.astype(bool)
        # valid trials: whether the orientation change will be on the
        # cued side
        self.is_trial_valid = ti.valid.astype(bool)
        # catch trials: no stimulus changes in the first 5 secs
        self.is_trial_catch = ti.catchtrial.astype(bool)
        # which side is the monkey cued to attend
        self.is_trial_cue_R = ti.attendloc.astype(bool)
        # which side the orientation change is on
        self.is_trial_target_R = ti.correctloc.astype(bool) & ~self.is_trial_catch
        # magnitude of orientation change
        self.target_orientation_change = ti.ochange.astype(float)
        self.target_orientation_change[ self.is_trial_catch ] = 0
        # end of trial code
        self.trial_end_code = ti.eotcode

        # block index
        trial_block_idx = np.cumsum(
                self.is_trial_cue_R[0][1:] != 
                self.is_trial_cue_R[0][:-1] )
        trial_block_idx = conc([ [0], trial_block_idx ])
        self.trial_block_idx = trial_block_idx

        # trial within the block
        self.trial_idx_within_block = conc([
            np.arange(np.sum(trial_block_idx == i)) 
            for i in range(np.max(trial_block_idx) + 1) ])
        
        """ How each trial played out """
       
        # how many stimuli presented
        ns = ti.numstim.copy()
        ns[np.isnan(ns)] = 0
        self.trial_N_stimuli_presented = ns.astype(int)
        
        # times in ms
        self.t_msec_absolute_trial_start = ti.trialstart
        self.t_msec_absolute_stimulus_on = ti.stimon
        #self.t_msec_absolute_fixation_cue_on = ti.fixon
        #self.t_msec_absolute_fixation_attained = ti.fixate                
        #self.t_msec_absolute_stimulus_off = ti.stimoff
        #self.t_msec_absolute_target_on = ti.targontime
        #self.t_msec_absolute_saccade_initiated = ti.saccade
        #self.t_msec_relative_fixate_to_target = ti.timetotarg
        #self.t_msec_relative_target_to_saccade = ti.rt
        
        """ Stimulus parameters """
        
        if hasattr( ti, 'gabor0' ):
            g0 = ti.gabor0
            self.gabor_R_azimuth_deg = g0.azimuthDeg        
            self.gabor_R_elevation_deg = g0.elevationDeg
            self.gabor_R_orientation_deg = g0.directionDeg
            self.gabor_R_spatial_freq_cpd = g0.spatialFreqCPD
            self.gabor_R_radius_deg = g0.radiusDeg
            self.gabor_R_sigma_deg = g0.sigmaDeg        
            self.gabor_R_misc = g0.__dict__.copy()
            
            g1 = ti.gabor1
            self.gabor_L_azimuth_deg = g1.azimuthDeg        
            self.gabor_L_elevation_deg = g1.elevationDeg
            self.gabor_L_orientation_deg = g1.directionDeg
            self.gabor_L_spatial_freq_cpd = g1.spatialFreqCPD
            self.gabor_L_radius_deg = g1.radiusDeg
            self.gabor_L_sigma_deg = g1.sigmaDeg        
            self.gabor_L_misc = g1.__dict__.copy()
            
            for k in ['_fieldnames', 'azimuthDeg', 'elevationDeg', 
                    'directionDeg', 'spatialFreqCPD', 'radiusDeg', 'sigmaDeg']:
                del self.gabor_R_misc[k]
                del self.gabor_L_misc[k]
            
        """ Units """
        
        self.unit_array_id = times['suarrayid']
        self.unit_names = A([ str(u).lower() for u in times['sulist'] ])

        # error check
        #assert times['numsu'] == len( self.unit_names ) == self.N_units

        """ Trials """

        # construct trial objects
        #from IPython.core.debugger import Tracer; Tracer()() 
        #from IPython.core.debugger import Tracer; Tracer()() 
        #from IPython.core.debugger import Tracer; Tracer()() 
        self.trials = np.array([ 
            Trial( i, tt, 
                t_msec_absolute_trial_start = ti.trialstart[i],
                is_trial_instruction = self.is_trial_instruction[0][i],
                is_trial_valid = self.is_trial_valid[0][i],
                is_trial_catch = self.is_trial_catch[0][i],
                is_trial_cue_R = self.is_trial_cue_R[0][i],
                is_trial_target_R = self.is_trial_target_R[0][i],
                trial_end_code = self.trial_end_code[i],
                trial_block_idx = self.trial_block_idx[i],
                trial_idx_within_block = self.trial_idx_within_block[i], 
                N_stimuli_presented = self.trial_N_stimuli_presented[0][i],
                target_orientation_change = self.target_orientation_change[0][i] )
            for i, tt in enumerate(times['trialtimes']) ])
        
        # 'trialtimes': (615,)
        # 'allsutimes': (63,)
        # 'suarrayid': (63,)
        # 'sulist': (63,)
        # 'unitid': (63,),
        # 'unittagarray': (63,)
        
        """
        Questions: 
        - are these only SUs, or also MUs? how do we tell them apart?
        - is "suarrayid" the "arrayid" referred to in the dataformat 
            document? what are "sulist", "unitid", "unittagarray"?        
        """

    """ Filtering trials """

    def filter_trials( self, boolarray ):
        # input check
        assert len(boolarray) == len(self)
        # create copy
        dnew = self.__class__( self.prefix, copydict=self.__dict__ )
        # whittle down the attributes
        attrs_to_whittle = [
                'is_trial_instruction', 'is_trial_valid', 'is_trial_catch',
                'is_trial_cue_R', 'is_trial_target_R', 
                'trial_end_code', 'trial_N_stimuli_presented', 
                'trial_block_idx', 'trial_idx_within_block',
                't_msec_absolute_trial_start', 't_msec_absolute_stimulus_on',
                'trials', 'target_orientation_change' ]
        for a in attrs_to_whittle:
            try:
                dnew.__dict__[a] = dnew.__dict__[a][boolarray]
            except TypeError:
                dnew.__dict__[a] = A( dnew.__dict__[a] )[boolarray]
        # return
        return dnew

    @property
    def filter_usable_trials( self ):
        return self.filter_trials( self.is_trial_usable )

    """ Filtering units """

    def filter_units( self, boolarray ):
        # input check
        assert len(boolarray) == self.N_units
        # create copy
        dnew = self.__class__( self.prefix, copydict=self.__dict__ )
        # whittle down the attributes
        attrs_to_whittle = [ 'unit_array_id', 'unit_names' ]
        for a in attrs_to_whittle:
            dnew.__dict__[a] = dnew.__dict__[a][boolarray]
        # recalculate trials
        dnew.trials = [ Trial(None, None, copydict=t.__dict__)
                for t in self.trials ]
        # for each trial
        for t in dnew.trials:
            t.t_spikes = t.t_spikes[boolarray]
            if hasattr(t, '_spike_counts'):
                del t._spike_counts
        # return
        return dnew

    """ Sugar """
    
    def __getitem__( self, i ):
        return self.trials[i]
        
    def __len__(self):
        return len(self.trials)

    def __repr__(self):
        return "<RawData '%s': %d units / %d trials>" % (
                self.prefix, self.N_units, self.N_trials )

    """ Useful properties: units """

    @property
    def N_units( self ):
        return len( self.unit_array_id )

    @property
    def is_unit_in_R_hemisphere( self ):
        return ( self.unit_array_id == 2 )

    @property
    def is_unit_in_L_hemisphere( self ):
        return ~self.is_unit_in_R_hemisphere

    @property
    def is_unit_rf_R( self ):
        return self.is_unit_in_L_hemisphere

    @property
    def is_unit_rf_L( self ):
        return self.is_unit_in_R_hemisphere
        
    """ Useful properties: trials """

    @property
    def N_trials( self ):
        return len(self)
        
    @property
    def is_trial_response_correct( self ):
        """ Correctly detected change, or fixated on catch trials. """
        return self.trial_end_code == 0

    @property
    def is_trial_response_missed( self ):
        """ A change happened and the monkey missed it. """
        return self.trial_end_code == 1

    @property
    def is_trial_response_early_cued( self ):
        """ False alarm to the cued stimulus. """
        return self.trial_end_code == 2

    @property
    def is_trial_response_early_uncued( self ):
        """ False alarm to the uncued stimulus. """
        return self.trial_end_code == 3

    @property
    def is_trial_response_broke_fixation( self ):
        """ Monkey saccaded to an non-stimulus location. """
        return self.trial_end_code == 4

    @property
    def is_trial_response_ignored( self ):
        """ Monkey never attained fixation. """
        return self.trial_end_code == 5
    
    @property
    def trial_response_type( self ):
        """ List of strings of what kind of response obtained. """
        types = {0:'correct', 1:'missed', 2:'early cued', 3:'early uncued',
                4:'broke', 5:'ignored'}
        return A([types[t] for t in self.trial_end_code])

    @property
    def is_trial_usable( self ):
        return ( ~self.is_trial_response_ignored 
                & ~self.is_trial_response_broke_fixation 
                & ~self.is_trial_instruction )

    @property
    def is_trial_cue_L( self ):
        return ~self.is_trial_cue_R

    @property
    def is_trial_target_L( self ):
        return ~self.is_trial_target_R & ~self.is_trial_catch

    @property
    def is_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.is_trial_target_L) | 
                (self.is_trial_cue_R & self.is_trial_target_R) )

    @property
    def is_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.is_trial_target_R) | 
                (self.is_trial_cue_R & self.is_trial_target_L) )

    """ Event data, as a separate data type """

    @property
    def all_event_data( self ):
        e = EventData( self.prefix )
        # event details
        from IPython.core.debugger import Tracer; Tracer()() 
        e.spike_counts = np.hstack([ t.spike_counts for t in self ])
        e.event_types = np.hstack([ t.event_types for t in self ])
        e.event_t_msec = np.hstack( self.t_msec_absolute_stimulus_on )
        e.event_durations = A( np.sum([[eb[1] - eb[0] for eb in t.event_bounds] 
            for t in self]) ).astype(int)
        # trial details
        e.trial_idxs = np.hstack([ 
            [t.trial_idx]*t.N_events for t in self ]).astype(int)
        e.trial_block_idxs = np.hstack([ 
            [t.trial_block_idx]*t.N_events for t in self ]).astype(int)
        e.trial_idxs_within_block = np.hstack([ 
            [t.trial_idx_within_block]*t.N_events for t in self ]).astype(int)
        e.is_trial_cue_R = A(
                np.sum([[t.is_trial_cue_R] * t.N_events for t in self])
                ).astype(bool)
        e.is_trial_usable = conc([ 
            [x]*t.N_events 
            for x, t in zip(self.is_trial_usable, self) ]).astype(bool)
        e.event_idxs_within_trial = conc([np.arange(t.N_events) for t in self])
        e.N_events_within_trial = conc([ 
            [t.N_events] * t.N_events for t in self ]).astype(int)
        # unit details
        e.unit_names = self.unit_names
        e.is_unit_rf_R = self.is_unit_rf_R
        # responses
        trial_end_code = self.trial_end_code
        prev_trial_end_code = conc([ [5], self.trial_end_code[:-1] ])
        prev_prev_trial_end_code = conc([ [5, 5], self.trial_end_code[:-2] ])
        e.trial_end_code = conc([ 
            [tec] * t.N_events 
            for tec, t in zip(trial_end_code, self) ]).astype(int)
        e.prev_trial_end_code = conc([ 
            [tec] * t.N_events 
            for tec, t in zip(prev_trial_end_code, self) ]).astype(int)
        e.prev_prev_trial_end_code = conc([ 
            [tec] * t.N_events 
            for tec, t in zip(prev_prev_trial_end_code, self) ]).astype(int)
        # what was the correct answer on each trial
        is_trial_target_R = self.is_trial_target_R
        prev_trial_target_R = conc([ 
            [False], self.is_trial_target_R[:-1] ]).astype(bool)
        prev_prev_trial_target_R = conc([ 
            [False, False], self.is_trial_target_R[:-2] ]).astype(bool)
        e.is_trial_target_R = conc([ 
            [tcr] * t.N_events 
            for tcr, t in zip(is_trial_target_R, self) ]).astype(bool)
        e.prev_trial_target_R = conc([ 
            [tcr] * t.N_events 
            for tcr, t in zip(prev_trial_target_R, self) ]).astype(bool)
        e.prev_prev_trial_target_R = conc([ 
            [tcr] * t.N_events 
            for tcr, t in zip(prev_prev_trial_target_R, self) ]).astype(bool)
        # catch
        is_trial_catch = self.is_trial_catch
        prev_trial_catch = conc([
            [True], is_trial_catch[:-1] ]).astype(bool)
        prev_prev_trial_catch = conc([
            [True, True], is_trial_catch[:-2] ]).astype(bool)
        e.is_trial_catch = conc([
            [x]*t.N_events 
            for x, t in zip(is_trial_catch, self) ]).astype(bool)
        e.prev_trial_catch = conc([
            [x]*t.N_events 
            for x, t in zip(prev_trial_catch, self) ]).astype(bool)
        e.prev_prev_trial_catch = conc([
            [x]*t.N_events 
            for x, t in zip(prev_prev_trial_catch, self) ]).astype(bool)
        # orientations
        e.target_orientation_change = conc([
            [toc] * t.N_events
            for toc, t in zip(self.target_orientation_change, self) ])
        # return
        return e

    @property
    def standard_event_data( self ):
        e = self.all_event_data
        to_keep = ( e.event_types == 'standard' ) & ( e.is_trial_usable )
        e = e.filter_events( to_keep )
        del e.event_types
        return e

class Trial( AutoCR ):

    """ Raw data for a single trial. """
    
    def __init__( self, idx, trialtimes, copydict=None, **kw ):
        # is this a copy?
        if copydict is not None:
            self.__dict__ = copydict.copy()
            return
        # parse input
        self.trial_idx = idx
        tt = trialtimes
        self.t_msec_relative_fixation_cue_on = tt.fixon
        self.t_msec_relative_fixation_attained = tt.fixate
        if isinstance( tt.stimon, np.ndarray ):
            self.t_msec_relative_stimulus_on = tt.stimon
        else:
            self.t_msec_relative_stimulus_on = A([ tt.stimon ])
        if isinstance( tt.stimoff, np.ndarray ):
            self.t_msec_relative_stimulus_off = tt.stimoff
        else:
            self.t_msec_relative_stimulus_off = A([ tt.stimoff ])
        self.t_msec_relative_target_on = tt.targon
        self.t_msec_relative_saccade_initiated = tt.saccade
        self.t_spikes = tt.sutimes
        # save keywords
        self.__dict__.update(**kw)

    """ Spike times """

    @property
    def N_units( self ):
        return len(self.t_spikes)

    @property
    def event_bounds( self ):
        t_on = self.t_msec_relative_stimulus_on
        t_off = self.t_msec_relative_stimulus_off
        if len(t_on) == 0:
            return []
        if len(t_on) == len(t_off) + 1:
            t_off = conc([t_off, 
                [self.t_msec_relative_saccade_initiated]])
        elif len(t_on) == len(t_off):
            pass
        else:
            print 't_on and t_off do not match'
            tracer()
        return zip( t_on, t_off )

    @property
    def event_types( self ):
        if self.N_stimuli_presented == 0:
            return []
        types = ['standard'] * self.N_stimuli_presented
        types[0] = 'first'
        if self.trial_response_type in ['early cued', 'early uncued']:
            t_stim_on = self.t_msec_relative_stimulus_on
            t_stim_off = self.t_msec_relative_stimulus_off 
            t_targ_on = self.t_msec_relative_target_on 
            N_stim = self.N_stimuli_presented 
            t_sacc = self.t_msec_relative_saccade_initiated
            if not np.isnan( t_targ_on ):
                print 'parse problem'
                tracer()
            if ( len(t_stim_off) < N_stim ):
                types[-1] = 'early'
            if np.isnan( t_sacc ):
                types[-1] = 'early'
            elif ( t_sacc - t_stim_on[-1] ) < 260:
                types = A(types)
                types[ t_sacc - t_stim_on < 260 ] = 'early'
                types = list(types)
        elif self.trial_response_type in ['missed', 'correct']:
            if not self.is_trial_catch:
                t_stim_on = self.t_msec_relative_stimulus_on
                t_targ_on = self.t_msec_relative_target_on 
                types = A(types)
                types[ t_stim_on == t_targ_on ] = 'target'
                types[ t_stim_on > t_targ_on ] = 'post'
                types = list(types)
        else:
            tracer()
        """
        elif self.trial_response_type == 'missed':
            types[-1] = 'target'
        elif self.trial_response_type == 'correct':
            if not self.is_trial_catch:
                types[-1] = 'target'
        """
        return types

    @property
    def N_events( self ):
        return len( self.t_msec_relative_stimulus_on )

    @property
    def event_bounds_for_spike_counts( self ):
        t_on = self.t_msec_relative_stimulus_on + 60
        t_off = self.t_msec_relative_stimulus_off + 60
        if len(t_on) == 0:
            return []
        if len(t_on) == len(t_off) + 1:
            t_off = conc([
                t_off, [self.t_msec_relative_saccade_initiated]])
        elif len(t_on) == len(t_off):
            pass
        else:
            print 't_on and t_off do not match'
            tracer()
        return zip( t_on, t_off )

    @property
    def _spike_count_durations_ms( self ):
        return A([ t[1]-t[0] 
            for t in self.event_bounds_for_spike_counts ]).astype(float)

    @property
    def spike_counts( self ):
        if not hasattr( self, '_spike_counts' ):
            bounds = self.event_bounds_for_spike_counts
            sc = self._spike_counts = np.zeros( 
                    (self.N_units, self.N_stimuli_presented), dtype=int )
            for i, t in enumerate( self.t_spikes ):
                for j, b in enumerate( bounds ):
                    sc[i, j] = np.sum( (t >= b[0]) & (t < b[1]) )
        return self._spike_counts

    """ Sugar """
        
    def __repr__( self ):
        return '<Trial %d>' % self.trial_idx

    @property
    def _dict_summary( self ):
        d = self.__dict__.copy()
        del d['_cache']
        del d['_proxies']
        # add extras
        extra_keys = ['trial_response_type', 'event_types']
        for k in extra_keys:
            d[k] = getattr( self, k )
        return self.__summarise__( d )

    """ Useful properties """

    @property
    def t_all( self ):
        return { k:v for k, v in self.__dict__.items() 
                if k.startswith('t_msec_relative') 
                or k.startswith('t_msec_absolute') }

    @property
    def trial_response_type( self ):
        """ String of what kind of response obtained. """
        return {0:'correct', 1:'missed', 2:'early cued', 3:'early uncued',
                4:'broke', 5:'ignored'}[self.trial_end_code]

    @property
    def is_trial_response_correct( self ):
        """ Correctly detected change, or fixated on catch trials. """
        return self.trial_end_code == 0

    @property
    def is_trial_response_missed( self ):
        """ A change happened and the monkey missed it. """
        return self.trial_end_code == 1

    @property
    def is_trial_response_early_cued( self ):
        """ False alarm to the cued stimulus. """
        return self.trial_end_code == 2

    @property
    def is_trial_response_early_uncued( self ):
        """ False alarm to the uncued stimulus. """
        return self.trial_end_code == 3

    @property
    def is_trial_response_broke_fixation( self ):
        """ Monkey saccaded to an non-stimulus location. """
        return self.trial_end_code == 4

    @property
    def is_trial_response_ignored( self ):
        """ Monkey never attained fixation. """
        return self.trial_end_code == 5
    
    @property
    def is_trial_cue_L( self ):
        return ~self.is_trial_cue_R

    @property
    def is_trial_target_L( self ):
        return ~self.is_trial_target_R & ~self.is_trial_catch

    @property
    def is_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.is_trial_target_L) | 
                (self.is_trial_cue_R & self.is_trial_target_R) )

    @property
    def is_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.is_trial_target_R) | 
                (self.is_trial_cue_R & self.is_trial_target_L) )

    @property
    def prev_trial_target_L( self ):
        return ~self.prev_trial_target_R & ~self.prev_trial_catch

    @property
    def prev_prev_trial_target_L( self ):
        return ~self.prev_prev_trial_target_R & ~self.prev_prev_trial_catch

    @property
    def prev_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_trial_target_L) | 
                (self.is_trial_cue_R & self.prev_trial_target_R) )

    @property
    def prev_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_trial_target_R) | 
                (self.is_trial_cue_R & self.prev_trial_target_L) )

    @property
    def prev_prev_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_prev_trial_target_L) | 
                (self.is_trial_cue_R & self.prev_prev_trial_target_R) )

    @property
    def prev_prev_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_prev_trial_target_R) | 
                (self.is_trial_cue_R & self.prev_prev_trial_target_L) )

"""
=================
EventsData class
=================
"""


class EventData( AutoCR ):

    """ Structure for storing data from events (rather than trials). """

    def __init__( self, prefix, empty_copy=False, **kw ):
        # save
        self.prefix = prefix
        self.__dict__.update( **kw )
        if empty_copy:
            return

    def __copy__( self ):
        """ Return a copy of self. """
        t = self.__class__( self.prefix, empty_copy=True )
        t.__dict__ = self.__dict__.copy()
        return t

    """ Properties """

    @property
    def N_units( self ):
        return self.spike_counts.shape[0]

    @property
    def N_events( self ):
        return self.spike_counts.shape[1]

    @property
    def Y__tn( self ):
        return self.spike_counts.T

    @property
    def is_unit_rf_L( self ):
        if self.is_unit_rf_R.dtype == bool:
            self.is_unit_rf_R = self.is_unit_rf_R.astype( bool )
        return ~self.is_unit_rf_R

    @property
    def is_trial_cue_L( self ):
        if self.is_trial_cue_R.dtype == bool:
            self.is_trial_cue_R = self.is_trial_cue_R.astype( bool )
        return ~self.is_trial_cue_R

    @property
    def S__ts( self ):
        return A([ self.is_trial_cue_L, self.is_trial_cue_R ]).T.astype(float)

    @property
    def is_trial_target_L( self ):
        if self.is_trial_target_R.dtype == bool:
            self.is_trial_target_R = self.is_trial_target_R.astype( bool )
        return ~self.is_trial_target_R & ~self.is_trial_catch

    @property
    def is_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.is_trial_target_L) | 
                (self.is_trial_cue_R & self.is_trial_target_R) )

    @property
    def is_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.is_trial_target_R) | 
                (self.is_trial_cue_R & self.is_trial_target_L) )

    @property
    def prev_trial_target_L( self ):
        return ~self.prev_trial_target_R & ~self.prev_trial_catch

    @property
    def prev_prev_trial_target_L( self ):
        return ~self.prev_prev_trial_target_R & ~self.prev_prev_trial_catch

    @property
    def prev_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_trial_target_L) | 
                (self.is_trial_cue_R & self.prev_trial_target_R) )

    @property
    def prev_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_trial_target_R) | 
                (self.is_trial_cue_R & self.prev_trial_target_L) )

    @property
    def prev_prev_trial_target_on_cued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_prev_trial_target_L) | 
                (self.is_trial_cue_R & self.prev_prev_trial_target_R) )

    @property
    def prev_prev_trial_target_on_uncued_side( self ):
        return ( (self.is_trial_cue_L & self.prev_prev_trial_target_R) | 
                (self.is_trial_cue_R & self.prev_prev_trial_target_L) )

    @property
    def block_idxs( self ):
        return set( self.trial_block_idxs )

    @property
    def N_block_idxs( self ):
        return len( self.block_idxs )

    @property
    def within_sequence_idx( self ):
        ti = self.trial_idxs
        within_seq_idx = np.zeros_like(ti) - 1
        for j in range(12):
            within_seq_idx[ ti == np.roll(ti, j) ] = j
        return within_seq_idx


    """ Filtering """

    _event_attributes_with_idx_last = {
            'event_durations', 'event_t_msec', 'is_trial_cue_R',
            'is_trial_usable', 'is_trial_catch', 'spike_counts', 'trial_block_idxs',
            'trial_idxs', 'trial_idxs_within_block', 'event_idxs_within_trial',
            'event_types', 'is_trial_target_R', 'trial_end_code',
            'prev_trial_target_R', 'prev_prev_trial_target_R',
            'prev_trial_end_code', 'prev_prev_trial_end_code',
            'prev_trial_catch', 'prev_prev_trial_catch',
            'N_events_within_trial', 'target_orientation_change'}

    _event_attributes_with_idx_first = {
            'SG__tn', 'S_global_soft__tn', 'sample__tn', 'Mu__tn',
            'SL__tn', 'SL_soft__tn' }

    _unit_attributes_with_idx_first = {
            'spike_counts', 'unit_names', 'is_unit_rf_R', 'K__ni', 'theta_h__ni',
            'K_soft__ni'}
    
    _unit_attributes_with_idx_last = {
            'SG__tn', 'S_global_soft__tn', 'sample__tn', 'Mu__tn', 'C__sn',
            'SL__tn', 'SL_soft__tn', 'A2__sn' }

    @property
    def _event_attributes( self ):
        return self._event_attributes_with_idx_first.union( 
                self._event_attributes_with_idx_last )

    @property
    def _unit_attributes( self ):
        return self._unit_attributes_with_idx_first.union(
                self._unit_attributes_with_idx_last )

    def filter_events( self, idxs ):
        """ Return a dataset with only the events specified by a bool array """
        t = self.__copy__()
        for k in self._event_attributes_with_idx_last:
            try:
                t.__dict__[k] = t.__dict__[k][..., idxs]
            except KeyError:
                pass
        for k in self._event_attributes_with_idx_first:
            try:
                t.__dict__[k] = t.__dict__[k][idxs, ...]
            except KeyError:
                pass
        # models
        models = [k for k,v in t.__dict__.items() 
                if ( k.startswith('C') or k.startswith('base_model') or k.startswith('mod_model') )
                and isinstance(v, Bunch)]
        for model in models:
            m = getattr( t, model ).copy()
            for k in self._event_attributes_with_idx_last:
                try:
                    m[k] = m[k][..., idxs]
                except KeyError:
                    pass
            for k in self._event_attributes_with_idx_first:
                try:
                    m[k] = m[k][idxs, ...]
                except KeyError:
                    pass
            setattr( t, model, m )
        return t

    def filter_units( self, idxs ):
        """ Return a dataset with only the units specified by a bool array """
        t = self.__copy__()
        for k in self._unit_attributes_with_idx_first:
            try:
                t.__dict__[k] = t.__dict__[k][idxs, ...]
            except KeyError:
                pass
        for k in self._unit_attributes_with_idx_last:
            try:
                t.__dict__[k] = t.__dict__[k][..., idxs]
            except KeyError:
                pass
        # models
        models = [k for k,v in t.__dict__.items()
                if ( k.startswith('C') or k.startswith('base_model') or k.startswith('mod_model') )
                and isinstance(v, Bunch)]
        for model in models:
            m = getattr( t, model ).copy()
            for k in self._unit_attributes_with_idx_first:
                try:
                    m[k] = m[k][idxs, ...]
                except KeyError:
                    pass
            for k in self._unit_attributes_with_idx_last:
                try:
                    m[k] = m[k][..., idxs]
                except KeyError:
                    pass
            setattr( t, model, m )
        return t

    @property
    def filter_rf_R( self ):
        return self.filter_units( self.is_unit_rf_R )

    @property
    def filter_rf_L( self ):
        return self.filter_units( self.is_unit_rf_L )

    @property
    def filter_cue_R( self ):
        return self.filter_events( self.is_trial_cue_R )

    @property
    def filter_cue_L( self ):
        return self.filter_events( self.is_trial_cue_L )

    def filter_to_block( self, block_idx ):
        if block_idx not in self.block_idxs:
            raise IndexError('no trials with block_idx = %d' % block_idx)
        return self.filter_events( self.trial_block_idxs == block_idx )

    @property
    def split_into_all_blocks( self ):
        return EventDataList([ 
            self.filter_to_block(b) for b in self.block_idxs ])

    @property
    def split_into_all_trials( self ):
        return EventDataList([
            self.filter_events(self.trial_idxs == t) 
            for t in np.unique(self.trial_idxs) ])

    def filter_to_unit( self, unit_idx ):
        if isinstance(unit_idx, int):
            to_keep = np.zeros( self.N_units, dtype=bool )
            to_keep[unit_idx] = True
            return self.filter_units( to_keep )
        elif isinstance(unit_idx, str):
            to_keep = (self.unit_names == unit_idx)
            if np.sum(to_keep) == 0:
                raise ValueError('no unit with that name')
            return self.filter_units( to_keep )
        else:
            raise ValueError('unit must be specified by an idx or name')

    @property
    def group_by_trial( self ):
        # unique trials
        trials, idxs = np.unique( self.trial_idxs, return_index=True )
        # new spike counts
        Y__tn = A([ self.spike_counts[ :, self.trial_idxs == t ].sum(axis=1) 
                for t in trials ])
        # how many events per trial
        l = list( self.trial_idxs )
        events_per_trial = A([ l.count(t) for t in trials ])
        # filter
        to_keep = np.zeros( self.N_events, dtype=bool )
        to_keep[idxs] = True
        d2 = self.filter_events( to_keep )
        d2.spike_counts = Y__tn.T
        d2.events_per_trial = events_per_trial
        d2.event_durations = A([ 
            self.event_durations[ self.trial_idxs == t ].sum() 
            for t in trials ])
        d2.mean_counts_per_event = (
            d2.spike_counts / d2.events_per_trial[na, :].astype(float) )
        return d2


    """ Sugar """

    def __repr__( self ):
        s = "<EventData '%s': %d units / %d events>" % (
                self.prefix, self.N_units, self.N_events )
        if self.N_units == 1:
            s = s.replace('units', 'unit')
        if self.N_events == 1:
            s = s.replace('events', 'event')
        return s

    """ Statistics """

    @property
    def mean( self ):
        return np.mean( self.spike_counts, axis=1 )

    @property
    def var( self ):
        return np.var( self.spike_counts, axis=1 )

    @property
    def fano( self ):
        return self.var / self.mean

    @property
    def noise_correlations( self ):
        return off_diag_elements( np.corrcoef( self.spike_counts ) )

    @property
    def noise_covariances( self ):
        return off_diag_elements( np.cov( self.spike_counts ) )

    """ Saving and loading attributes """

    def _attr_dir( self, attr ):
        return os.path.join( computations_dir, attr )

    def _attr_filename( self, attr ):
        save_prefix = self.prefix
        return os.path.join( self._attr_dir(attr), '%s.pickle' % save_prefix )

    def save_attribute( self, attr, overwrite=False, 
            include_event_idxs=True, include_unit_idxs=True ):
        """ Save a computed attribute to the `computations` subdirectory.

        The `overwrite` keyword determines what to do if the file already
        exists. If True, the file is overwritten. If False, an error is raised.
        If None, nothing happens.

        By default, indexes associated with the events and the units are 
        saved as well; this allows for matching when reloading. These indexes
        are provided via the fields `event_t_msec` and `unit_names`.
        
        """
        # filename
        target_dir = self._attr_dir( attr )
        filename = self._attr_filename( attr )
        # check if it exists
        if os.path.exists( filename ):
            if overwrite is True:
                pass
            elif overwrite is False:
                raise IOError('file `%s` exists' % filename )
            elif overwrite is None:
                return
            else:
                raise ValueError('`overwrite` should be True/False/None')
        # prepare output
        output = {}
        output[attr] = getattr( self, attr )
        if include_event_idxs:
            output['event_t_msec'] = self.event_t_msec
        elif attr in self._event_attributes:
            raise ValueError('must include event_t_msec for saving `%s`' % attr) 
        if include_unit_idxs:
            output['unit_names'] = self.unit_names
        elif attr in self._unit_attributes:
            raise ValueError('must include unit_idxs for saving `%s`' % attr) 
        # does the directory exist
        if not os.path.exists( target_dir ):
            try:
                os.mkdir( target_dir )
            except OSError:
                if not os.path.exists( target_dir ):
                    os.mkdir( target_dir )
                pass
        # save
        cPickle.dump( output, open(filename, 'w'), 2 )

    def can_load_attribute( self, attr ):
        """ Whether an attribute has been computed and saved to disk. """
        filename = self._attr_filename( attr )
        return os.path.exists(filename)

    @property
    def available_attributes( self ):
        return sorted( [ a for a in os.listdir(computations_dir) 
            if self.can_load_attribute(a) ] )

    def load_attribute( self, attr, overwrite=True, ignore_if_missing=False ):
        """ Load a computed attribute. """
        if isinstance( attr, list ):
            for a in attr:
                self.load_attribute( a, overwrite=overwrite )
            return
        if hasattr( self, attr ) and not overwrite:
            return
        # filename
        filename = self._attr_filename( attr )
        # load
        try:
            d = cPickle.load( open( filename ) )
        except IOError as e:
            if ignore_if_missing:
                return
            else:
                raise e
        a = d[attr]
        # filter trials
        if d.has_key( 'event_t_msec' ):
            # if the current dataset has any trials the saved attr does not
            if not in1d( self.event_t_msec, d['event_t_msec'] ).all():
                raise ValueError('saved dataset is missing trials')
            # if the saved attr has any trials the current dataset does not
            events_to_keep = in1d( d['event_t_msec'], self.event_t_msec )
            if not events_to_keep.all():
                if attr in self._event_attributes_with_idx_first:
                    a = a[ events_to_keep, ... ]
                elif attr in self._event_attributes_with_idx_last:
                    a = a[ ..., events_to_keep ]
                elif attr.startswith('C') or attr.startswith('base_model') or attr.startswith('mod_model'):
                    print {k:v.shape for k, v in a.items()}
                    for k in self._event_attributes_with_idx_last:
                        try:
                            a[k] = a[k][..., idxs]
                        except KeyError:
                            pass
                    for k in self._event_attributes_with_idx_first:
                        try:
                            a[k] = a[k][idxs, ...]
                        except KeyError:
                            pass
                    print {k:v.shape for k, v in a.items()}
                    tracer()
                elif isinstance( attr, np.ndarray ):
                    err_str = 'attr `%s` needs to be added to filtering list'
                    raise ValueError( err_str % attr )
        elif attr in self._event_attributes:
            raise ValueError('no `event_t_msec` included in saved `%s`' % attr) 
        # filter cells
        if d.has_key( 'unit_names' ):
            # if the current dataset has any cells the saved attr does not
            if not in1d( self.unit_names, d['unit_names'] ).all():
                raise ValueError('saved dataset is missing units')
            # if the saved attr has any units the current dataset does not
            cells_to_keep = in1d( d['unit_names'], self.unit_names )
            if not cells_to_keep.all():
                if attr in self._unit_attributes_with_idx_first:
                    a = a[ cells_to_keep, ... ]
                elif attr in self._unit_attributes_with_idx_last:
                    a = a[ ..., cells_to_keep ]
                elif attr.startswith('C') or attr.startswith('base_model') or attr.startswith('mod_model'):
                    print {k:v.shape for k, v in a.items()}
                    for k in self._unit_attributes_with_idx_first:
                        try:
                            a[k] = a[k][ cells_to_keep, ... ]
                        except KeyError:
                            pass
                    for k in self._unit_attributes_with_idx_last:
                        try:
                            a[k] = a[k][ ..., cells_to_keep ]
                        except KeyError:
                            pass
                    print {k:v.shape for k, v in a.items()}
                elif isinstance( attr, np.ndarray ):
                    raise ValueError(
                            'attr `%s` needs to be added to filtering list')
        elif attr in self._unit_attributes:
            raise ValueError('no `unit_names` included in saved `%s`' % attr) 
        # save
        setattr( self, attr, a )


    """
    =====
    ADMM
    =====
    """
    
    def solve_naive( self, **kw ):
        s = models.Naive( self, **kw )
        return Bunch({ 
            'Omega__tn':s.Omega__tn, 'nu__n':s.nu__n, 'rank':s.rank, 'tau':s.tau,
            'LL_per_obs':s.LL_per_obs })

    def solve_cue( self, **kw ):
        s = models.Cue( self, **kw )
        return Bunch({ 
            'Omega__tn':s.Omega__tn, 'Nu__sn':s.Nu__sn, 'rank':s.rank, 'tau':s.tau,
            'LL_per_obs':s.LL_per_obs, 'S__ts':s.S__ts })

    def solve_seq( self, **kw ):
        s = models.Seq( self, **kw )
        return Bunch({ 
            'Omega__tn':s.Omega__tn, 'Nu__sn':s.Nu__sn, 'rank':s.rank, 'tau':s.tau,
            'LL_per_obs':s.LL_per_obs, 'S__ts':s.S__ts })

    def _parse_hemi_kw( self, G__tn=None, C__sn=None, **kw ):
        """ Helper function. """
        # keywords
        if [ G__tn, C__sn ].count(None) == 0:
            kwL = kw.copy()
            kwR = kw.copy()
            if isinstance(G__tn, int) or isinstance(G__tn, float):
                kwL['G__tn'] = G__tn
                kwR['G__tn'] = G__tn
            elif G__tn is not None:
                kwL['G__tn'] = G__tn[ :, self.is_unit_rf_L ]
                kwR['G__tn'] = G__tn[ :, self.is_unit_rf_R ]
            if C__sn is not None:
                kwL['C__sn'] = C__sn[ :, self.is_unit_rf_L ]
                kwR['C__sn'] = C__sn[ :, self.is_unit_rf_R ]
        elif [ G__tn, C__sn ].count(None) == 1:
            raise ValueError('must provide neither or both G__tn/C__sn')
        else:
            kwL = kwR = kw
        return kwL, kwR

    def _parse_hemi_results( self, mL, mR ):
        """ Helper function. """
        m = Bunch({})
        m.Omega__tn = np.zeros( (self.N_events, self.N_units) )
        m.Omega__tn[ :, self.is_unit_rf_L ] = mL.Omega__tn
        m.Omega__tn[ :, self.is_unit_rf_R ] = mR.Omega__tn
        m.rank = mL.rank
        m.tau_L = mL.tau
        m.tau_R = mR.tau
        m.LL_per_obs = ( 
                (mL.LL_per_obs * self.is_unit_rf_L.sum()) + 
                (mR.LL_per_obs * self.is_unit_rf_R.sum()) ) / self.N_units
        return m

    def solve_naive_hemi( self, tau_L=1, tau_R=1, G__tn=None, C__sn=None, **kw ):
        # keywords
        kwL, kwR = self._parse_hemi_kw( G__tn=G__tn, C__sn=C__sn, **kw )
        # solve separately
        mL = self.filter_rf_L.solve_naive( tau=tau_L, **kwL )
        mR = self.filter_rf_R.solve_naive( tau=tau_R, **kwR )
        # put together
        m = self._parse_hemi_results( mL, mR )
        m.nu__n = np.zeros(self.N_units) * np.nan
        m.nu__n[ self.is_unit_rf_L ] = mL.nu__n
        m.nu__n[ self.is_unit_rf_R ] = mR.nu__n
        return Bunch({ 'all':m, 'L':mL, 'R':mR })

    def solve_cue_hemi( self, tau_L=1, tau_R=1, G__tn=None, C__sn=None, **kw ):
        # keywords
        kwL, kwR = self._parse_hemi_kw( G__tn=G__tn, C__sn=C__sn, **kw )
        # solve separately
        mL = self.filter_rf_L.solve_cue( tau=tau_L, **kwL )
        mR = self.filter_rf_R.solve_cue( tau=tau_R, **kwR )
        # put together
        m = Bunch({})
        m = self._parse_hemi_results( mL, mR )
        m.Nu__sn = np.zeros( (2,self.N_units) ) * np.nan
        m.Nu__sn[ :, self.is_unit_rf_L ] = mL.Nu__sn
        m.Nu__sn[ :, self.is_unit_rf_R ] = mR.Nu__sn
        m.S__ts = mL.S__ts
        return Bunch({ 'all':m, 'L':mL, 'R':mR })

    def solve_seq_hemi( self, tau_L=1, tau_R=1, G__tn=None, C__sn=None, **kw ):
        # keywords
        kwL, kwR = self._parse_hemi_kw( G__tn=G__tn, C__sn=C__sn, **kw )
        # solve separately
        mL = self.filter_rf_L.solve_seq( tau=tau_L, **kwL )
        mR = self.filter_rf_R.solve_seq( tau=tau_R, **kwR )
        # put together
        m = Bunch({})
        m = self._parse_hemi_results( mL, mR )
        m.Nu__sn = np.zeros( (22, self.N_units) ) * np.nan
        m.Nu__sn[ :, self.is_unit_rf_L ] = mL.Nu__sn
        m.Nu__sn[ :, self.is_unit_rf_R ] = mR.Nu__sn
        m.S__ts = mL.S__ts
        return Bunch({ 'all':m, 'L':mL, 'R':mR })


"""
=================
All data together
=================
"""

class SuperDataList( AutoCR ):

    """ Superclass for collection of data. """

    def __init__(self, *data):
        if len(data) == 1 and np.iterable(data):
            self._data = data[0]
        else:
            self._data = list( data )

    def __getitem__( self, key ):
        if isinstance(key, int):
            return self._data[key]
        elif isinstance(key, slice):
            data = self._data[key]
            return RawDataList(data)
        elif isinstance( key, str ) or isinstance( key, unicode ):
            data = [ d for d in self._data if (d.prefix == key) ] 
            if len(data) == 1:
                return data[0]
            elif len(data) == 0:
                raise IndexError('no dataset with that prefix')
            else:
                return RawDataList( *data )

    def __setitem__( self, key, val ):
        if isinstance( key, int ):
            self._data[key] = val
        elif isinstance( key, str ) or instance( key, unicode ):
            idx = np.nonzero([
                (d.prefix == key) for d in self._data ])[0]
            if len(idx) == 1:
                self._data[ idx[0] ] = val
            elif len(idx) == 0:
                raise IndexError('no Data with label `%s`' % key)
            else:
                raise IndexError('more than one Data with label `%s`' % key)
        else:
            raise TypeError('unknown key type')

    def __delitem__( self, key ):
        if isinstance( key, int ):
            self._data.__delitem__( key )
        elif isinstance( key, str ) or instance( key, unicode ):
            idx = np.nonzero([
                (d.prefix == key) for d in self._data ])[0]
            if len(idx) == 1:
                self._data.__delitem__( idx[0] )
            elif len(idx) == 0:
                raise IndexError('no Data with label `%s`' % key)
            else:
                raise IndexError('more than one Data with label `%s`' % key)
        else:
            raise TypeError('unknown key type')

    def __len__( self ):
        return len(self._data)

    @property
    def N_units( self ):
        return sum([ d.N_units for d in self ])

    def __add__( self, tl2 ):
        return self.__class__( self._data + tl2._data )

    def __radd__( self, tl1 ):
        return self.__class__( tl1._data + self._data )

    def __iter__( self, *a, **kw ):
        return self._data.__iter__( *a, **kw )

    def append( self, d ):
        self._data.append(d)
    

class RawDataList( SuperDataList ):

    """ Nice collection of raw data. """

    @property
    def N_trials( self ):
        return sum([ d.N_trials for d in self ])

    @property
    def filter_usable_trials( self ):
        """ Return a DataList including only the usable trials. """
        return self.__class__([ d.filter_usable_trials for d in self._data ])

    def __repr__( self ):
        return '<%d RawData sets: total %d units, %d trials>' % (
                len(self), self.N_units, self.N_trials )


class EventDataList( SuperDataList ):

    """ Nice collection of event data. """

    @property
    def N_events( self ):
        return sum([ d.N_events for d in self ])

    def __repr__( self ):
        return '<%d EventData sets: total %d units, %d events>' % (
                len(self), self.N_units, self.N_events )

    """ Saving and loading attributes """

    def save_attribute( self, attr, **kw ):
        """ Save the computed attribute for all."""
        for d in self:
            d.save_attribute( attr, **kw )

    def load_attribute( self, attr, overwrite=True, ignore_if_missing=False ):
        """ Load the computed attribute for all."""
        for d in self:
            d.load_attribute( attr, overwrite=overwrite, 
                    ignore_if_missing=ignore_if_missing )

    def filter_by_attribute( self, attr ):
        return self.__class__([ d for d in self._data 
            if hasattr( d, attr ) or attr in d.available_attributes ])

    @property
    def available_attributes( self ):
        attrs = sorted( os.listdir(computations_dir) )
        attr_len = A([ len(a) for a in attrs ]).max()
        for a in attrs:
            s = ' ' * (attr_len + 2 - len(a))
            s += a
            s += '  :  % 2d' % len(self.filter_by_attribute(a))
            print s



"""
=========
Mop model
=========
"""


class MopData( mop.Data ):
    
    def __init__( self, X, y ):
        super( MopData, self ).__init__( X, y, add_constant=False )

    @property
    def k0( self ):
        ym = A([ 
            ( self.y[ self.X[:, i] > 0 ].mean() 
                / self.X[self.X[:, i] > 0, i].mean() )
            for i in range(self.D) ])
        return np.log(ym)

        
class MopSolver( mop.ML_k, mop.Lowpass_h, mop.UnitSolver ):

    @property
    def _max_len_h_star( self ):
        return min([ 1000, self.data.T ])
    
    @cached
    def training_idxs( data ):
        return np.isfinite( data.y )
    
    @cached
    def has_testing_regions():
        return False
    
    @cached
    def T_training( training_idxs ):
        return training_idxs.sum()
    
    @cached
    def T_testing():
        return 0
    
    def _slice_by_testing( self, sig ):
        return A([])
    
    def _slice_by_training( self, sig ):
        return sig[ self.training_idxs, ... ]
    
    def _zero_during_testing( self, sig ):
        sig = sig.copy()
        sig[ ~self.training_idxs, ... ] = 0
        return sig


"""
===========================
Helpful interface to models
===========================
"""

class BaseModel( Bunch ):

    @property
    def G__tn( self ):
        if self.has_key('SG__tn') and self.has_key('SL__tn'):
            return np.exp( self.SG__tn + self.SL__tn )
        elif self.has_key('SG__tn'):
            return np.exp( self.SG__tn )
        elif self.has_key('SL__tn'):
            return np.exp( self.SL__tn )
        else:
            return 1

    


class Model( AutoCR ):
    
    def __init__( self, result ):
        # save result
        self.__dict__.update( **result )
        # svd
        USV = svd( self.Omega__tn, full_matrices=False )
        S__ti = USV[0][:, :self.rank ]
        W__in = USV[2][ :self.rank, : ] * USV[1][ :self.rank ][:, na]
        # rescale
        scale_i = S__ti.std( axis=0 )
        S__ti /= scale_i[na, :]
        W__in *= scale_i[:, na]
        # save
        self.S__ti = S__ti
        self.W__in = W__in
        # flip weights
        for i in range(self.rank):
            if self.W__in[i, :].mean() < 0:
                self.S__ti[:, i] *= -1
                self.W__in[i, :] *= -1
        # rank
        if self.rank == 1:
            self.s__t = self.S__ti[ :, 0 ]
            self.w__n = self.W__in[ 0, : ]
        
    def __repr__( self ):
        d = self.__dict__.copy()
        del d['_cache']
        del d['_proxies']
        return Bunch(d).__repr__()

"""
=========
Load data
=========
"""

def process_data( save_raw=False, save_events=True ):
    # load the raw data
    raw_data = RawDataList([ 
        RawData(p) for p in progress.dots( prefixes, 'loading raw data' ) ])
    # only include cells with mean rates > 0.1 
    raw_data = RawDataList([
        d.filter_units( d.all_event_data.Y__tn.mean( axis=0 ) > 0.1 )
        for d in progress.dots( raw_data, 'filtering out silent cells' ) ])
    # load the event data
    event_data = EventDataList( [
        d.filter_usable_trials.standard_event_data 
        for d in progress.dots(raw_data, 'compiling event data') ])
    # save it
    if save_raw:
        filename = join( data_dir, 'raw_data.pickle' )
        cPickle.dump( raw_data, open(filename, 'w'), 2 )
    if save_events:
        filename = join( data_dir, 'event_data.pickle' )
        cPickle.dump( event_data, open(filename, 'w'), 2 )
    return {'raw_data':raw_data, 'event_data':event_data}

