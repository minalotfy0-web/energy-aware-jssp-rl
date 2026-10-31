
import numpy as np
from petrirl.envs.moo.petri_build import Petri_build
from petrirl.render.graph import  Graph
from petrirl.utils.obs import get_obs_moo
from petrirl.utils.twin import create_twin
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", message=".*env.action_masks to get variables from other wrappers is deprecated.*")



class Simulator(Petri_build):
    """
    Class representing the core logic of a Job Shop Scheduling Problem (JSSP) simulation using a Petri net.

    Attributes:
        clock (int): The internal clock of the simulation.
        interaction_counter (int): Counter for interactions in the simulation.
        delivery_history (dict): Dictionary storing the delivery history.
        action_map (dict): Mapping for actions in the simulation from discrete to multidiscrete.

    Methods:
        __init__(instance_id, dynamic=False, size=(None,None), n_agv=2, n_tt=1):
            Initializes the JSSP simulator with optional parameters.

        petri_reset():
            Resets the internal state of the Petri net, including places and tokens.

        action_mapping():
            Maps multidiscrete actions to a dictionary for use with reinforcement learning.

        is_terminal():
            Checks if the simulation has reached a terminal state.

        sort_tokens():
            Processes tokens in sorting places based on their roles.

        refresh_state():
            Updates the state of the Petri net after sorting tokens and checking transitions.

        fire_timed():
            Advances simulation time and fires timed transitions based on elapsed times.

        action_masks():
            Checks which transitions are enabled for action selection.

        fire_controlled(action):
            Fires a controlled transition based on the provided action.

        time_tick():
            Increments the internal simulation clock and updates token logging.

        interact(action, screenshot=False):
            Performs Petri net interactions based on the action and updates internal state.
    """

    def __init__(self, 
                 label="main",
                 benchmark='Solax',
                 instance_id="sl01", 
                 layout=1,
                 n_agv=1,
                 n_tt=0,
                 usage_cap=0.2,
                 standby=True,
                 dynamic=False,
                 size=(None,None,None,None),
                 
                 observation_depth=1,
                 lookup=False
                  ):
        """
    Initializes the JSSP simulator with the given parameters.

    Parameters:
        instance_id (str): Identifier for the JSSP instance.
        dynamic (bool): If True, allows appending new operations; termination condition is all queues empty.
        size (tuple): Size parameters for simulation only if dynamic  (default: (None, None)).
        n_agv (int): Number of AGVs (default: 2).
        n_tt (int): Number of tool transports (default: 1).
    """
    
        super().__init__(instance_id, 
                         layout= layout,
                         benchmark=benchmark,
                         n_agv=n_agv,
                         n_tt=n_tt,
                         # usage_cap=usage_cap,
                         standby= standby,
                         dynamic=dynamic,
                         size=size,
                         )
        
        
        self.label=label
        self.create_petri(show_flags=False,show_develop=False) 
        self.cache =self.cache_nodes()


        self.clock = 0
        self.interaction_counter = 0
        self.interaction_timing=[]
        
        
        self.external_mask={}
        self.observation_depth=observation_depth
        
        self.delivery_history = {}
        self.action_map = self.action_mapping()
        self.graph=Graph(self)
        
        self.lookup=lookup
        if self.lookup :
            self.twin=create_twin(self)
            
        self.feedbacks={}
        #self.graph.plot_net()
        
        self.usage_cap = usage_cap
        
        np.random.seed(101)
        self.machines_powers = np.random.randint(10, 100, size=self.n_machines)
        self.target_consumption =self.usage_cap *   self.machines_powers.sum()
        
        self.lower_limit=0
        self.upper_limit=self.machines_powers.sum()
        self.logs ={
           "energy_consumption":[0],
           "machine_utilization":[],
           "energy_overshoot":[],
           "utilization_overshoot":[],
            }
        
        
        
        self.petri_reset()
        self.print_instance_info()
        
        
    def plot_energy_consumption(self):
        """Plot the energy consumption on the main axes and display the plot."""
        # Generate x-axis values
        x_values = np.arange(0, len(self.logs["energy_consumption"]))
       
        target_values = np.full_like(x_values, self.target_consumption)
        min_values = np.full_like(x_values, min(self.machines_powers))
        max_values = np.full_like(x_values, sum(self.machines_powers))
    
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, max_values, color='red', linewidth=3, label="Max Consumption")
        plt.plot(x_values, target_values, color='green', linewidth=2, label="Target Consumption")
        plt.plot(x_values, min_values, color='red', linewidth=3, label="Min Consumption")
        plt.plot(x_values, self.logs["energy_consumption"], color='blue', linewidth=2, label="Actual Consumption")
    
        # Customize the plot
        plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=1)
        plt.ylabel('Energy Consumption (Power unit)', fontsize=18)
        plt.xlabel('Time (units)', fontsize=18)
        #plt.legend(fontsize=16)
        plt.title('Energy Consumption Over Time', fontsize=16)
    
        # Show the plot
        plt.show()
        
        

 
    def get_utilization(self):
         
         utilization = 0
         busy_machines= np.array( [bool(machine.token_container) for machine in self.cache["machine_processing"]])
         utilization =  sum(busy_machines)/self.n_machines
         
         self.logs["machine_utilization"].append(utilization)
         
         
         if "utilization_overshoot" not in self.logs:
             self.logs["utilization_overshoot"] = []  # Reinitialize if missing
        
         utilization_overshoot = (utilization-self.usage_cap)/self.usage_cap
        
         self.logs["utilization_overshoot"].append(utilization_overshoot)
         
         return utilization   
     
        
    def get_consumption(self):
        
        consumption = 0
        busy_machines= np.array( [bool(machine.token_container) for machine in self.cache["machine_processing"]])
        consumption =  np.dot(busy_machines , self.machines_powers)
        
        # print (consumption)
        
        energy_overshoot = (consumption - self.target_consumption)/self.target_consumption
        
        
        self.logs["energy_consumption"].append(consumption)
        if "energy_overshoot" not in self.logs:
            self.logs["energy_overshoot"] = []  # Initialize the key with an empty list
        self.logs["energy_overshoot"].append(energy_overshoot)
        
        return consumption
    
            
    def set_test_mode(self,model):  
        if self.lookup:  
            self.twin.policy=model.policy
    
        
    def print_instance_info (self): 
        if self.label=="main":
            # print(f"{self.benchmark} Instance '{self.instance_id}' ,Layout: '{self.layout}' is loaded.")
            # print(f"JSSP {self.instance_id}: {self.n_jobs} jobs X {self.n_machines} machines X  {self.n_tools} Tools, AGVs: {self.n_agv}, TT: {self.n_tt}, Dynamic Mode: {self.dynamic} , Lookup: {self.lookup} ,Standby: {self.standby}")
            pass
    def get_state(self):
        
        return get_obs_moo(self,self.observation_depth)
     
        
        
    def petri_reset(self):
        """
        Resets the internal state of the Petri net, including the simulation clock and token containers in places.
        """
        
        #self.plot_energy_consumption()
 
        self.clock = 0
        self.logs ={
           "energy_consumption":[],
           "machine_utilization":[]
            }
        
    
        for place in self.places.values():
            place.token_container = []
            
        self.add_tokens()

    def action_mapping(self):
         """
         Maps multidiscrete actions to a more versatile Discrete format to use with exp DQN.
         Returns:
             dict: Mapping dictionary from action number  to transition object .
         """
         controllabe_transitions =[t for t in self.transitions.values() if t.type == "c"]
         mapping_dict = {index: value for index, value in enumerate(controllabe_transitions)}
         return mapping_dict


    def is_terminal(self):
        """Checks if the simulation has reached a terminal state."""
                
        return np.all([len(p.token_container) == 0 for p in self.cache["flow_places"]])
    
    def time_tick(self):
        """Advances the internal clock and updates token logging."""
        
        self.clock += 1
        self.feedbacks['energy_consumption']=self.get_consumption()
        self.feedbacks['machine_utilisation']=self.get_utilization()
        for p in self.cache["process_places"]:
            p.tick()
       
      
    def fire_auto(self):
         """Fires automatic transitions."""
         
         for transition in self.cache["auto_transitions"]: 
             if transition.check_state():
                 transition.fire(self.clock)
             
             
    def sort_tokens(self):
         """Sorts tokens in sorting places based on their roles."""
         
         
         def process_tokens(place, color_criterion_index):
             tokens = place.token_container
             if not tokens:
                 return
             
             for token in tokens.copy():
                 for transition in place.children:
                     if token.color[color_criterion_index] == transition.color:
                         transition.fire(self.clock)
                         break
                 else:
                     tokens.remove(token)
                     print(f"Token color: {token.color} destroyed in {place.role} - No compatible destination found!")
     
         role_to_index = {
             "job_sorting": 0,
             "machine_sorting": 1,
             "machine_sorting_T": 1,
             "request_sorting": 2,
             "tools_sorting": 2,
         }
     
         for place in self.cache.get("sorting_places", []):
                 process_tokens(place, role_to_index[place.role])
                 
                 
    def refresh_state(self):
         """Refreshes the state of the Petri net after sorting tokens and checking enabled transitions."""
         self.sort_tokens()
         self.fire_auto()
                

    

    def fire_timed(self):
        """
       Advances the simulation time, fires timed transitions, and updates the state of the Petri net.
       Returns:
           list: List of UIDs of transitions that fired.
       """
       
        role_dict={"machine_processing":0,
                   "agv_transporting":1,
                   "agv_relocating":2,
                   "tt_transporting":3,
                   "tt_relocating":4,
                   }
       
        
        fired_transitions = []
        
        for place in (p for p in self.cache["process_places"] if p.type == "p" and p.token_container):
            
            transition = place.children[0]
            token = place.token_container[0]   
            
            elapsed_time = token.logging[place.uid][2]
            reference_time = token.time_features[role_dict[place.role]]

            if elapsed_time >= reference_time :
                
                if place.role in ["agv_transporting", "tt_transporting"]:
                    self.get_deadheadings(place,role_dict)
                    
                transition.fire(self.clock)
                fired_transitions.append(transition.uid)
                
                
        self.refresh_state()
        self.time_tick()   
        
        self.delivery_history[self.clock] = [ token for place in self.places.values() if place.type == "d" for token in place.token_container]
        return fired_transitions
           
        

    def fire_controlled(self, action):
        """
       Fires a colored transition based on the provided action if it is enabled.
       Parameters:
           action: Action to be performed.
       Returns:
           list: List containing the UID of the fired transition, or an empty list if no transition was fired.
       """
        fired_transitions=[]  
        
        transition = self.action_map[int(action)] 
        
        # if transition.label=="standby":
        self.fire_timed()
              
        transition.fire(self.clock) 
        fired_transitions.append(transition.uid)
        
        self.interaction_counter += 1  
        return fired_transitions
    
    
    
    def guard_function(self):
        """Refreshes the state of the Petri net after sorting tokens and checking enabled transitions."""

        self.refresh_state()
        mask =[t.check_state()  for t in self.transitions.values() if t.type == "c"]
        return mask  



    def interact(self, action ,screenshot):
        """
        Performs Petri net interactions based on the provided action and updates the internal state.
    
        Parameters:
            action: Action to be performed.
            screenshot (bool): If True, generates a plot of the Petri net after each interaction (default: False).
        """
        
        
        # self.feedbacks['energy_consumption']=self.get_consumption()
        # self.feedbacks['machine_utilisation']=self.get_utilization()
        self.interaction_timing.append(self.clock)
        
        fired_controlled = self.fire_controlled(action)
        
        # self.graph.plot_net(fired_controlled) if screenshot else None
        
        
        # while sum(self.guard_function()) == 1:
            
            
        #     fired_timed = self.fire_timed()
            
            
        #     self.graph.plot_net(fired_timed) if screenshot else None
        # #     # self.feedbacks['energy_consumption']=self.get_consumption()
        # #     # self.feedbacks['machine_utilisation']=self.get_utilization()
        #     if self.is_terminal():
        #         self.feedbacks['energy_consumption']=self.get_consumption()
        #         self.feedbacks['machine_utilisation']=self.get_utilization()
        #         break
            
        # self.feedbacks['energy_consumption']=self.get_consumption()
        # self.feedbacks['machine_utilisation']=self.get_utilization()
        #self.feedbacks['valid_move']=True
        
        
        return self.feedbacks 
           
            

if __name__ == "__main__":
    
    
    benchmark='Raj'
    instance = "ra01"
    layout =1
    n_agv=0
    n_tt=0
    usage_cap=0.2
    standby=True
    dynamic=False
    # Job,machines,n_agvs,n_tools,ntt
    size=(100,20,10,0,0)
    
    petri = Simulator(benchmark=benchmark,
                      instance_id= instance ,
                      layout=layout,
                      n_agv=n_agv,
                      n_tt=n_tt,
                      usage_cap=usage_cap,
                      standby=standby,
                      dynamic=dynamic , 
                      size= size) 
    
    petri.graph.plot_net()
    petri.print_instance_info()
    
    print(petri.standby)   
    print(petri.machines_powers)
    
 