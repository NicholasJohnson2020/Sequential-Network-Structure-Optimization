import numpy as np

import networkx as nx
import scipy.sparse as sparse

# Base class for environment model
class ObesityModel():

    def __init__(self, initial_network, target_nodes, healthy_nodes,
                 lamda_vec, exog_info_fn_name, objective_fn_name,
                 T=100, alpha = 0.01, beta = 1.0001, mu = 0.1):
        
        # Validate exog_info_fn_name input parameter
        assert (exog_info_fn_name == 'Uniform'
               )|(exog_info_fn_name == 'Weighted')
        
        # Validate objective_fn_name input parameter
        assert (objective_fn_name == 'Cumulative'
               )|(objective_fn_name == 'Terminal')
        
        state_vec = np.reshape(np.asarray(list(nx.get_node_attributes(
            initial_network, name='state').values())), newshape = [-1,1])
        degree = [v for u, v in initial_network.in_degree()]
        
        decision_set = []
        for u in healthy_nodes:
            for v in target_nodes:
                if not initial_network.has_edge(u,v):
                    decision_set.append((u,v))
        
        self.initial_state = {'Network': initial_network.copy(),
                              'Nodes': state_vec,
                              'Size': initial_network.number_of_nodes(),
                              'Targets': target_nodes,
                              'Volunteers': healthy_nodes,
                              'Lambda Vector': lamda_vec}
        
        # Initialize model state
        self.state = {'Network': initial_network.copy(),
                      'Nodes': state_vec,
                      'Step': 0,
                      'Decision Set': decision_set}
        
        # Initialize model parameters
        self.params = {'T': T,
                       'Alpha': alpha,
                       'Beta': beta,
                       'Mu': mu,
                       'Degree': degree,
                       'Lambda': sparse.diags(lamda_vec, format='csr')}
        
        self.exog_info_fn_name = exog_info_fn_name
        self.objective_fn_name = objective_fn_name
        
        # Initialize objective function
        self.objective = 0.0
        
    # This function gives the exogeneous information
    def exog_info_fn(self, decision):

        chi_in = np.zeros(self.initial_state['Size'])
        for (u, v) in decision:
            chi_in[v] = chi_in[v] + 1
        
        if self.exog_info_fn_name == 'Uniform':
            return self.uniform_exog_info(chi_in)
        elif self.exog_info_fn_name == 'Weighted':
            return self.weighted_exog_info(chi_in)
        else:
            return None
    
    # This function gives the exogeneous information under
    # uniform edge removals.
    def uniform_exog_info(self, chi):
        
        edge_removals = []
        
        for i in range(self.initial_state['Size']):
            if chi[i] == 0:
                continue
            edges = list(self.state['Network'].in_edges([i]))
            new_removals = [edges[i] 
                            for i in np.random.choice(np.arange(len(edges)),
                                                      int(chi[i]),
                                                      replace = False)]
            edge_removals = edge_removals + new_removals
        
        return edge_removals
    
    # This function gives the exogeneous information under
    # weighted edge removals.
    def weighted_exog_info(self, chi):
        
        edge_removals = []
        
        for i in range(self.initial_state['Size']):
            if chi[i] == 0:
                continue
            edges = list(self.state['Network'].in_edges([i], data = True))
            prob = [1 - d['weight'] for (u,v,d) in edges]
            norm = sum(prob)
            prob = [p/norm for p in prob]
            new_removals = [edges[i][0:2]
                            for i in np.random.choice(np.arange(len(edges)),
                                                      int(chi[i]),
                                                      replace = False,
                                                      p = prob)]
            edge_removals = edge_removals + new_removals
        
        return edge_removals

    # This function executes the model transition dynamics
    def transition_fn(self, decision, exog_info):
        
        old_network = self.state['Network']
        new_network = old_network.copy()
        
        # Update the graph edges
        for u, v in exog_info:
            new_network.remove_edge(u, v)
            if (u in self.initial_state['Volunteers']
               )&(v in self.initial_state['Targets']):
                self.state['Decision Set'].append((u,v))
        for u, v in decision:
            new_network.add_edge(u, v, weight=self.params['Degree'][v]**-2)
            self.state['Decision Set'].remove((u,v))
        
        # Update the edge weights
        chi_in = np.zeros(self.initial_state['Size'])
        for (u, v) in decision:
            chi_in[v] = chi_in[v] + 1

        for i in range(self.initial_state['Size']):
            if chi_in[i] == 0:
                for u, v, d in new_network.in_edges([i], data=True):
                    new_weight = (1-self.params['Mu'])*d['weight']
                    new_weight += self.params['Mu']/self.params['Degree'][i]
                    d['weight'] = new_weight
            else:
                # Normalize edge weights at nodes where edges
                # were added and removed
                weight_total = 0
                in_edges = new_network.in_edges([i], data=True)
                for u, v, d in in_edges:
                    weight_total += d['weight']
                for u, v, d in in_edges:
                    d['weight'] = d['weight']/weight_total

        # Update the node states
        adj = nx.adjacency_matrix(new_network).transpose()
        identity = sparse.identity(self.initial_state['Size'], format='csr')
        factor = identity - self.params['Lambda']
        factor = factor + self.params['Lambda'].dot(adj)
        new_state = factor.dot(self.state['Nodes'])
        self.state['Nodes'] = new_state           
                    
        self.state['Network'] = new_network
        
        # Increment step count
        self.state['Step'] = self.state['Step'] + 1
        
        return 

    # This function updates model objective
    def objective_fn(self, decision):
        
        if self.objective_fn_name == 'Cumulative':
            return self.cumulative_objective_fn(decision=decision)
        elif self.objective_fn_name == 'Terminal':
            return self.terminal_objective_fn(decision=decision)
        else:
            return None

    # The cumulative objective function
    def cumulative_objective_fn(self, decision):
        
        penalty = np.power(self.params['Beta'],
                           self.state['Step'])*len(decision)
        self.objective -= self.params['Alpha']*penalty
        
        for i in self.initial_state['Targets']:
            self.objective += self.state['Nodes'][i]
    
    # The terminal objective function
    def terminal_objective_fn(self, decision):
        
        penalty = np.power(self.params['Beta'],
                           self.state['Step'])*len(decision)
        self.objective -= self.params['Alpha']*penalty
        
        if self.state['Step'] == self.params['T']:
            for i in self.initial_state['Targets']:
                self.objective += self.state['Nodes'][i]
    
    # This function step the process forward by one time increment
    def step(self, decision, debug=False):

        exog_info = self.exog_info_fn(decision)
        self.transition_fn(decision, exog_info)
        if debug:
            self.test_integrity()
        self.objective_fn(decision)
        
    # Returns the health status of the target nodes
    def get_target_health(self):
    
        health = [self.state['Nodes'][v]
                  for v in self.initial_state['Targets']]
        return health
    
    # Returns the health status of the volunteer nodes
    def get_volunteer_health(self):
        
        health = [self.state['Nodes'][v] 
                  for v in self.initial_state['Volunteers']]
        return health
    
    # Returns the health status of all nodes
    def get_all_health(self):
        
        return self.state['Nodes']
    
    # Checks invariants of the state representation
    def test_integrity(self):
        
        # Check number of nodes
        assert self.state['Network'].number_of_nodes() == self.initial_state['Size']
        
        # Check degree of each node
        degree = [v for u, v in self.state['Network'].in_degree()]
        assert degree == self.params['Degree']
        
        # Check edge weight sums and edge weight range
        for node in self.state['Network'].nodes():
            weight_total = 0
            in_edges = self.state['Network'].in_edges([node], data = True)
            for u, v, d in in_edges:
                assert (0 <= d['weight'])&(d['weight'] <= 1)
                weight_total += d['weight']
            assert round(weight_total, 10) == 1
            
        # Check state range
        for state in self.state['Nodes']:
            assert (0 <= state)&(state <= 1)

