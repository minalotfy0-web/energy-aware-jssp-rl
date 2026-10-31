import numpy as np

def get_obs_basic(places, observation_depth=1):
    observation = []

    for place in places:
        if place.token_container:
            token = place.token_container[0] 
            
            elapsed = token.logging.get(place.uid, [0, 0, 0])[2]  
            
            # Append values in one step for efficiency
            observation += [
                len(place.token_container),        # x1
                token.rank,                        # x1
                int(token.last_op),                # x1
                *token.color,                      # x3
                *token.machine_sequence,           # x3
                *token.time_features,              # x5
                token.time_features[1] - elapsed   # x1
            ]
        else:
            observation += [0] * 15 
            
    return np.array(observation, dtype=np.int64)


def get_obs_dft(places,mask,observation_depth=1):
    
    observation = list (get_obs_basic(places))+mask
    return np.array(observation, dtype=np.int64)



def get_obs_moo(sim,observation_depth):
   
    observation = list (get_obs_basic(sim.places.values(), observation_depth))
    
    machines = [p for p in sim.places.values() if p.uid in sim.filter_nodes("machine_processing") ]
    
    #get total consumption
    consumption=0
    busy_machines= np.array( [bool(machine.token_container) for machine in machines])
    consumption =  np.dot(busy_machines , sim.machines_powers)
    observation.append(consumption)
    # observation.append(sim.clock)
    
    #Get consumption per machine:       
    for idx,machine in enumerate(machines):
       observation.append( bool(machine.token_container) * sim.machines_powers[idx])
                               

    return np.array(observation, dtype=np.int64)



