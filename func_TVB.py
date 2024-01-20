from tvb.simulator.lab import *
import numpy as np

def LB_DconnParams():
    LB_params=dict(variables_of_interest=['V','W','Z'], 
                   aei=np.array(2.),  aee=np.array(0.35), aie=np.array(2.0), ane=np.array(1), ani=np.array(0.42),
                   b=np.array(0.1), 
                   C=np.array(0.1),
                   d_Ca=np.array(0.15),d_Na=np.array(0.15),d_K=np.array(0.4),d_V=np.array(0.65), d_Z=np.array(0.6), 
                   Iext=np.array(0.3), 
                   gCa=np.array(1.1), gK=np.array(2.5), gL=np.array(0.7),gNa=np.array(6.7),
                   phi=np.array(0.7),
                   QV_max=np.array(1), QZ_max=np.array(1),
                   rNMDA=np.array(0.25),
                   tau_K=np.array(1),  TCa=np.array(-0.01),TNa=np.array(0.26),TK=np.array(0.0), 
                   VCa=np.array(1), VK=np.array(-0.7), VL=np.array(-0.5), VNa=np.array(0.53), VT=np.array(-0.1),
                   ZT=np.array(0.))
    return LB_params

def LB_HconnParams():
    LB_params=dict(variables_of_interest=['V','W','Z'], 
               aei=np.array(2.0),  aee=np.array(0.5), aie=np.array(2.0), ane=np.array(1), ani=np.array(0.4), #ane=1
               b=np.array(0.1), 
               C=np.array(0.1),
               d_Ca=np.array(0.15),d_Na=np.array(0.15),d_K=np.array(0.4),d_V=np.array(0.65), d_Z=np.array(0.6), 
               Iext=np.array(0.3),
               gCa=np.array(1.1), gK=np.array(2.5), gL=np.array(1.1),gNa=np.array(6.7), 
               phi=np.array(0.7),
               QV_max=np.array(1), QZ_max=np.array(1),
               rNMDA=np.array(0.25),
               tau_K=np.array(1),  TCa=np.array(-0.01),TNa=np.array(0.26),TK=np.array(0.0), 
               VCa=np.array(1.1), VK=np.array(-0.7), VL=np.array(-0.5), VNa=np.array(0.53), VT=np.array(-0.1),
               ZT=np.array(0))
    return LB_params

def run_sim(conn, params,  dt=0.1, 
            manipulated_nodes=[], manipulation_params=dict(),
            fit_nodes=[], fit_params=dict(),
            stim_onset = 100., stim_T = 1e15, stim_tau = 10, stim_amp = 0,  stim_node=13,
            sim_length = 1e3, 
            integ = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([1e-7])))):

    # CONNECTIVITY:
    conn.configure()

    # SIMULATION COMPONENTS:
    mon = (monitors.TemporalAverage(period=1.0),) #
    coupl=coupling.HyperbolicTangent(a=np.array(0.5 * params['QV_max']),           
                                     midpoint=params['VT'],        
                                     sigma=params['d_V'] )

    initial_data = np.zeros((1, 3, conn.number_of_regions, 1)) 
 
    for node in range(conn.number_of_regions):
        initial_data[:,0,node,0] = np.random.uniform(-0.1,0.1) 
        initial_data[:,1,node,0] = np.random.uniform(-0.1,0.1) 
        initial_data[:,2,node,0] = np.random.uniform(-0.1,0.1) 
        
    #PARAMS
    if len(fit_nodes)>0:
        for param in fit_params:
            params[param] = np.ones((conn.number_of_regions, ))*params[param]
            params[param][fit_nodes] = fit_params[param]
                
    if len(manipulated_nodes)>0 and manipulated_nodes[0] is not None:
        for param in manipulation_params:
            params[param] = np.ones((conn.number_of_regions, ))*params[param]
            params[param][manipulated_nodes] = manipulation_params[param]
        
    lb=models.LarterBreakspear(**params)
    
    # STIMULUS:
    weighting = np.zeros((conn.number_of_regions, )) 
    weighting[stim_node] = 1
    eqn_t = equations.PulseTrain()
    eqn_t.parameters['onset'] = stim_onset
    eqn_t.parameters['tau'] = stim_tau
    eqn_t.parameters['T'] = stim_T
    eqn_t.parameters['amp'] = stim_amp
    stim = patterns.StimuliRegion(temporal=eqn_t,
                                  connectivity=conn,
                                  weight=weighting)
    # SIMULATION:
    sim = simulator.Simulator(integrator = integ, connectivity = conn,
                              coupling=coupl, model = lb, monitors = mon, 
                              stimulus = stim, initial_conditions = initial_data)
    sim.configure()
                                           
    # RESULTS:
    result = sim.run(simulation_length = sim_length)
    time = result[0][0]
    data = np.squeeze(result[0][1])
    data_V = data[:,0]
    data_W = data[:,1]
    data_Z = data[:,2]
    
    return time, data, data_V, data_W, data_Z 

def NearestNodes(connectivity, central_node, start, end,  k=3):
    
    '''
        Find the k regions closest (Euclidean distances) to a given node (central_node).
        
    start: index of the first region of the connectome
    end:   index of the last region of the connectome
    k:     number of the closest regions to be obtained
        
        To get the closest regions within a hemisphere "start" and "end" must be the start 
        and end indices of the desired hemisphere (e.g., right hemisphere in Dconn: start = 0, end = 38).
    '''
    node = np.array(central_node)
        
    # Coordinates of the central node
    xo = connectivity.centres[node, 0]
    yo = connectivity.centres[node, 1]
    zo = connectivity.centres[node, 2]

    # Compute distances 
    nor = connectivity.number_of_regions
    distances = np.sqrt((connectivity.centres[start:end, 0] - xo)**2 
                +  (connectivity.centres[start:end, 1] - yo)**2 
                +  (connectivity.centres[start:end, 2] - zo)**2)

    # get closest nodes
    if end==connectivity.weights.shape[0]:
        sorted_dist = np.argsort(distances) + 38 # indices of the left hemisphere
    else:
        sorted_dist = np.argsort(distances)

    nodes_indices = sorted_dist[:k+1]   

    return nodes_indices
