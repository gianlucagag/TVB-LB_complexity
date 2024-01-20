import numpy as np

def spike_detect(t,volt,HardThreshold=-0.05,ThresholdTime=0.2,dt=0.1):
    '''
    '''
    global ind,ind2,ind3,indN
    idxThresholdTime=int(ThresholdTime/dt)
    #print(idxThresholdTime)
    ind=np.where(volt>HardThreshold)[0]
    ind=np.hstack((ind,1e10))
    ind=np.array(ind,'int')
    ind2=np.diff(ind) #ind[1:]-ind[:-1] 
    ind3=np.where(ind2>idxThresholdTime)[0]
    indN=ind[ind3]        
    return t[indN]

def calc_firing(data, time, bin_size=25, start=3000, end=4000, stim_onset=3500, sliding=True, step=10, trial=0):
    ''' 
    Calculate the instantaneous firing rate (IFR) in Hz for the selected trial.

    Parameters:
    - data:       Time series data, with dimensions nTrials x nTime x nNodes or nTime x nNodes (if 1 trial).
    - time:       Time values corresponding to the time series data.
    - bin_size:   Bin size in milliseconds for computing IFR.
    - start:      Starting time for the analysis. 
    - end:        Final time for the analysis. 
    - stim_onset: Stimulus onset time. The stimulus will be at time 0. Set to 0 if no stimulation. 
    - sliding:    If True, use a sliding window for IFR computation. Default is True.
    - step:       Step size for the sliding window if 'sliding' is True. Default is 10 ms.
    - trial:      Selected trial index. Set to 0 if there is only one trial. Default is 0.

    Returns:
    - tuple: A tuple containing two arrays.
        - IFR (array): Instantaneous firing rates in Hz, with dimensions nAreas x nTimeBins.
        - TimeBins (array): Time bins corresponding to the computed firing rates.
    '''
    
    nSpikes, SpikeCount = [],[]
    
    # nTime x nNodes --> 1 x nTime x nNodes: for when there is only one trial (trial 0)  
    if len(data.shape) == 2:
        data = np.expand_dims(data,axis=0)

    for node in range(data.shape[2]): 
        i = start 
        
        if sliding == True:
            TimeBins = np.arange(start-stim_onset, (end-bin_size)-stim_onset, step)
            while (i+bin_size) < end: 
                spikes = spike_detect(time[i:i+bin_size], data[trial][i:i+bin_size,node])
                nSpikes.append(len(spikes))
                i += step
            SpikeCount.append(nSpikes)
            nSpikes = []
            
        else:     
            TimeBins = np.arange(start,end,bin_size)  
            while i<end: 
                spikes = spike_detect(time[i:i+bin_size], data[trial][i:i+bin_size,node])
                nSpikes.append(len(spikes))
                i += bin_size
            SpikeCount.append([nSpikes])
            nSpikes = []
            
    '''  
        # faster version
        if sliding == True:
            TimeBins = np.arange(start,end-bin_size,step)
            all_spikes = spike_detect(time, data[start:end,node])
            for t in TimeBins:
                spikes = np.where((all_spikes>=t) & (all_spikes<t+bin_size))[0]
                nSpikes.append(len(spikes))
            SpikeCount.append(nSpikes)
            nSpikes = []

        else:
            TimeBins = np.arange(start,end,bin_size)
            all_spikes = spike_detect(time, data[start:end,node])
            for t in TimeBins:
                spikes = np.where((all_spikes>=t) & (all_spikes<t+bin_size))[0]
                nSpikes.append(len(spikes))
            SpikeCount.append(nSpikes)
            nSpikes = []
    '''
  
    SpikeCount = np.squeeze(SpikeCount)
    
    # Instantaneous firing rate (Hz)
    IFR = SpikeCount/(bin_size/1000) 
    
    return IFR, TimeBins

def calc_firing_trials(data, time, bin_size, start, end, stim_onset, conn, sliding=True, step=10,):
    '''
    Calculate the instantaneous firing rate (IFR) in Hz on all trials

    Parameters:
    - data:       Time series data, with dimensions nTrials x nTime x nNodes.
    - time:       Time values corresponding to the time series data.
    - bin_size:   Bin size in milliseconds for computing firing rates.
    - start:      Starting time for the analysis. 
    - end:        Final time for the analysis. 
    - stim_onset: Stimulus onset time. The stimulus will be at time 0. Set to 0 if no stimulation.
    - conn:       Connectivity matrix.
    - sliding:    If True, use a sliding window for firing rate computation. Default is True.
    - step:       Step size for the sliding window if 'sliding' is True. Default is 10 ms.
  
    Returns:
    - tuple: A tuple containing two arrays.
        - IFR_trials (array): Instantaneous firing rates in Hz, with dimensions nTrials x nAreas x nTimeBins.
        - TimeBins (array): Time bins corresponding to the computed firing rates.
    '''
    
    nTrials, nTime, nNodes = data.shape
    
    IFR_trials = []
    for trial in range(nTrials): 
        IFR, TimeBins = calc_firing(data=data, time=time, trial=trial, bin_size=bin_size, start=start,
                           end=end, stim_onset=stim_onset, sliding=sliding, step=step)

        # Replace 0 values (of manipulated nodes) to avoid division by 0 in the statistic
        zero_sum_nodes = np.sum(IFR[:, :len(IFR[0])//2], axis=1) == 0
        zero_sum_weights = np.sum(conn, axis=1) == 0 # also get rid of non-connected nodes
        mask = zero_sum_nodes | zero_sum_weights
        IFR[mask, :] = np.random.uniform(1e-12, 1e-16, size=IFR[mask, :].shape)
        IFR_trials.append(IFR)
        
    IFR_trials = np.array(IFR_trials)
    
    return IFR_trials, TimeBins

def DictSpikesRegion(data, start=500, end=4500, node=13, ht=-0.05, bin_size=25,stim_onset=1000, All_Trials=True, kTrial=0): 
    ''' 
    Create a dictionary containing the trials and the respective spikes for the selected node if ALL_Trials=True,
    otherwise dictionary containing the selected trial and the respective spikes for the selected node
        
    data:       nTrials x nTime x nAreas
    start:      Starting time
    end:        Final time
    node:       Idx of the area of interest
    ht:         Hard threshold for the spike detection
    bin_size:   Bin size in ms for the spike count
    stim_onset: Stimulus onset time 
    kTrial:     Selected trial (if All_Trial=False)
    '''
    
    # nTime x nNodes --> 1 x nTime x nNodes: for when there is only one trial (trial 0)  
    if len(data.shape) == 2:
        data = np.expand_dims(data,axis=0)
    
    SpikesTrials={}
    TotSpikes, SpikeCount, nSpikes = [],[],[]
   
    time = np.arange(-stim_onset, data.shape[1]-stim_onset)
    
    
    if All_Trials==True:
        # Dictionary with all trials and respective spikes (timing)
        for trial in range(data.shape[0]):    
            t,v=time[start:end],data[trial][start:end,node]
            tk=spike_detect(t,v,HardThreshold=ht)
            SpikesTrials.update({'trial {}'.format(trial+1):tk})

        # Total spikes sorted by timing
        for trial in range(1, data.shape[0]+1):
            for spike in SpikesTrials['trial {}'.format(trial)]:
                TotSpikes.append(spike)
        TotSpikes=np.sort(TotSpikes)
        
    else:
        # Dictionary with selected trial and respective spikes (timing)
        for trial in range(kTrial-1,kTrial):
            t,v=time[start:end],data[trial][start:end,node]
            tk=spike_detect(t,v,HardThreshold=ht)
            SpikesTrials.update({'trial {}'.format(trial+1):tk})
        
        # Total spikes sorted by timing
        for trial in range(kTrial,kTrial+1): 
            for spike in SpikesTrials['trial {}'.format(trial)]:
                TotSpikes.append(spike)
        TotSpikes=np.sort(TotSpikes)
    
    # Spikes count in the bin
    if len(TotSpikes)>0:
        start_bin=-stim_onset+start
        end_bin=start_bin+bin_size           
        while end_bin<=(-stim_onset+end): 
            for s in TotSpikes:
                if start_bin<=s<end_bin:
                    nSpikes.append(s)
            SpikeCount.append(len(nSpikes))
            nSpikes = []
            start_bin = end_bin
            end_bin = end_bin + bin_size
        TimeBins = np.arange(start-stim_onset, end-stim_onset, bin_size)

    else:
        TimeBins=np.arange(-stim_onset+start,-stim_onset+end,bin_size) 
        for t in TimeBins:
            SpikeCount.append(0)
    
    return SpikesTrials, SpikeCount, TimeBins