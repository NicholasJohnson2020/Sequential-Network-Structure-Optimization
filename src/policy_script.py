# Import relevant packages
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.preprocessing import normalize
import scipy.sparse as sparse

# Evaluate the terminal objective value after taking decision X
def evaluate_function(model, X):

    state = model.state
    targets = model.initial_state['Targets']
    params = model.params

    dim = model.initial_state['Size']
    ones = np.zeros((1,dim))
    for i in targets:
        ones[0,i] = 1

    state_level = np.reshape(state['Nodes'], newshape=(-1,1))
    W = nx.adjacency_matrix(state['Network'], weight=None).transpose()
    temp = normalize(W + X, norm='l1', axis=1).todense()
    identity = np.identity(dim)
    lamb = params['Lambda'].todense()

    phi = np.dot(lamb, temp) + identity - lamb

    result = np.dot(np.linalg.matrix_power(phi, params['T']), state_level)
    result = np.dot(ones, result)

    return result[0,0]

# Sample uniform edge removals after adding the edges in the list decision
def uniform(model, decision):

    edge_removals = []

    chi = np.zeros(model.initial_state['Size'])
    for (u, v) in decision:
        chi[v] = chi[v] + 1

    for i in range(model.initial_state['Size']):
        if chi[i] == 0:
            continue
        edges = list(model.state['Network'].in_edges([i]))
        new_removals = [edges[i]
                for i in np.random.choice(np.arange(len(edges)),
                                          int(chi[i]),
                                          replace = False)]
        edge_removals = edge_removals + new_removals

    return edge_removals

# Sample weighted edge removals after adding the edges in the list decision
def weighted(model, decision):

    edge_removals = []

    chi = np.zeros(model.initial_state['Size'])
    for (u, v) in decision:
        chi[v] = chi[v] + 1

    for i in range(model.initial_state['Size']):
        if chi[i] == 0:
            continue
        edges = list(model.state['Network'].in_edges([i], data = True))
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

# Control Policy - take no action
def control_policy(model):

    return []

# Random Policy - reutrn a random set of edges
def random_policy(model):

    state = model.state
    healthy = model.initial_state['Volunteers']
    degree = model.params['Degree']

    data_struct = {}
    for h in healthy:
        data_struct[h] = []

    # Sort the candidate edges by healthy node
    for (u,v) in state['Decision Set']:
        data_struct[u].append(v)

    degree_count = np.zeros(model.initial_state['Size'],
                            dtype=int)
    decision_set = []

    # For each healthy node, add an edge chosen at random
    # to the decision set provided it does not violate
    # the decision constraint
    for h in healthy:
        if len(data_struct[h]) == 0:
            continue
        target = np.random.choice(data_struct[h], size=1)[0]
        if degree_count[target] < degree[target]:
            decision_set.append((h, target))
            degree_count[target] = degree_count[target]+1

    return decision_set

# Execute the Random Policy if t=0
def random_wrapper_policy(model):

    if model.state['Step'] == 0:
        return random_policy(model)
    else:
        return []

# Optimized Myopic Policy designed for Cumulative Objective
def optimized_myopic_policy(model):

    state = model.state
    targets = model.initial_state['Targets']
    params = model.params
    mode = model.exog_info_fn_name

    # Validate mode input parameter
    assert (mode == 'Uniform'
           )|(mode == 'Weighted')

    decision_set = []
    data_struct = {}
    mem_struct = {}

    for v in targets:
        data_struct[v] = {}
    threshold = params['Alpha']*np.power(params['Beta'], state['Step'])

    adj = nx.adjacency_matrix(state['Network']).transpose()
    diag = np.asarray([i**-1 for i in params['Degree']])
    weight_factor = sparse.diags(params['Mu']*diag, format='csr')
    adj_unweighted = nx.adjacency_matrix(state['Network'],
                                         weight=None).transpose()
    weights = (1-params['Mu'])*adj+weight_factor.dot(adj_unweighted)

    losses = weights.dot(state['Nodes'])

    delta = adj.dot(state['Nodes'])

    # ITerate through candidate edges
    for (u,v) in state['Decision Set']:

        const = mem_struct.get(v, -1)

        if const == -1:

            edges = state['Network'].in_edges([v], data = True)
            val_1 = 0
            val_2 = 0

            for (w,x,d) in edges:
                gamma = 1+(params['Degree'][v]**2)*(1-d['weight'])
                if mode == 'Uniform':
                    prob = 1/params['Degree'][v]
                else:
                    prob = (1-d['weight'])/(params['Degree'][v]-1)
                factor = delta[v]-d['weight']*state['Nodes'][w]
                factor = factor*(params['Degree'][v]**2)
                val_1 = val_1 + prob/gamma
                val_2 = val_2 + (prob/gamma)*factor

            const = [val_1, val_2]
            mem_struct[v] = const

        total = state['Nodes'][u]*const[0]+const[1]
        gain = total - losses[v]
        if (gain > threshold/params['Lambda'][v,v]):
            data_struct[v][(u, v)] = gain

    for key in data_struct:
        counter = Counter(data_struct[key])
        high = counter.most_common(params['Degree'][key])
        for i in high:
            decision_set.append(i[0])

    return decision_set

# Optimized Simple Lookahead Policy for Cumulative Objective
def optimized_simple_lookahead_policy(model):

    state = model.state
    targets = model.initial_state['Targets']
    params = model.params
    mode = model.exog_info_fn_name

    # Validate mode input parameter
    assert (mode == 'Uniform'
           )|(mode == 'Weighted')

    decision_set = []
    data_struct = {}
    mem_struct = {}

    # Initialize threshold (cost) values
    for v in targets:
        data_struct[v] = {}
    threshold = params['Alpha']*np.power(params['Beta'], state['Step'])

    # Calculate two-step objective value assuming no action
    mu = params['Mu']
    adj = nx.adjacency_matrix(state['Network']).transpose()
    diag = np.asarray([i**-1 for i in params['Degree']])
    weight_factor = sparse.diags(mu*diag, format='csr')
    adj_unweighted = nx.adjacency_matrix(state['Network'],
                                         weight=None).transpose()
    weights = (1-mu)*adj+weight_factor.dot(adj_unweighted)
    losses_0 = weights.dot(state['Nodes'])

    identity = sparse.identity(model.initial_state['Size'], format='csr')
    factor = identity - params['Lambda']
    factor = factor + params['Lambda'].dot(adj)
    states_1 = factor.dot(state['Nodes'])
    weights = (1-mu)*adj+weight_factor.dot(adj_unweighted)
    losses_1 = weights.dot(states_1)

    mult = (2*identity-params['Lambda']).dot(params['Lambda'])
    total_losses = mult.dot(losses_0)+params['Lambda'].dot(losses_1)

    # Calculate values needed for algorithm
    delta = adj.dot(state['Nodes'])
    state_1_w_sum = np.reshape(adj.dot(states_1),(-1,))
    state_1_sum = np.reshape(adj_unweighted.dot(states_1),(-1,))
    scaling = np.asarray([i**2 for i in params['Degree']])
    theta = (1-mu)*np.multiply(scaling, state_1_w_sum)
    pi = mu*np.multiply(diag, state_1_sum)

    # Iterate through candidate edges
    for (u,v) in state['Decision Set']:

        lamb = params['Lambda'][v,v]

        const = mem_struct.get(v, -1)
        if const == -1:

            edges = state['Network'].in_edges([v], data = True)
            vals = np.zeros(4)

            for (w,x,d) in edges:
                gamma = 1+(params['Degree'][v]**2)*(1-d['weight'])
                if mode == 'Uniform':
                    prob = 1/params['Degree'][v]
                else:
                    prob = (1-d['weight'])/(params['Degree'][v]-1)

                vals[0] = vals[0]+prob/gamma
                factor = delta[v]-d['weight']*state['Nodes'][w]
                factor = factor*(params['Degree'][v]**2)
                vals[1] = vals[1]+(prob/gamma)*factor
                factor = (1-mu)*params['Degree'][v]+mu*gamma
                vals[2] = vals[2]+(prob/gamma)*(factor/params['Degree'][v])
                factor = (1-mu)*(params['Degree'][v]**2)*d['weight']
                factor = (factor+(mu*gamma/params['Degree'][v]))*states_1[w]
                factor = -1*factor+theta[v]+gamma*pi[v]
                vals[3] = vals[3] + (prob/gamma)*factor

            const = list(vals)
            mem_struct[v] = const

        total_0 = state['Nodes'][u]*const[0]+const[1]
        total_1 = states_1[u]*const[2]+const[3]
        gain = (2-lamb)*(total_0)+total_1
        gain = lamb*gain - total_losses[v]
        if (gain > threshold):
            data_struct[v][(u, v)] = gain
    for key in data_struct:
        counter = Counter(data_struct[key])
        high = counter.most_common(params['Degree'][key])
        for i in high:
            decision_set.append(i[0])

    return decision_set

# Executes the simple lookahead policy defined above
# only on even steps
def optimized_simple_lookahead_wrapper_policy(model):

    if model.state['Step'] % 2 == 0:
        return optimized_simple_lookahead_policy(model)
    else:
        return []

# Optimized Gradient Based Policy inspired by
# the RECONNECT algorithm
def optimized_gradient_policy(model, iterations=10, samples=10):

    mode = model.exog_info_fn_name

    # Validate mode input parameter
    assert (mode == 'Uniform'
           )|(mode == 'Weighted')

    decision_set = []
    values = np.zeros(iterations)

    # Initialize the algorithm
    gradient, _ = gradient_oracle(model=model,
                                  decision=[],
                                  iterations=1)
    decision_t = linear_oracle(model=model,
                               gradient=gradient)
    decision_set.append(decision_t)

    # Main loop
    for i in range(iterations):
        # Perform Monte Carlo gradient and function value estimate
        gradient, score = gradient_oracle(model=model,
                                          decision=decision_set[i],
                                          iterations=samples)
        values[i] = score
        # Perform the linear optimization step
        decision_t = linear_oracle(model=model,
                                   gradient=gradient)
        decision_set.append(decision_t)

    return decision_set[np.argmax(values)]

# Calculates a Monte Carlo Gradient and Function value estimate
def gradient_oracle(model, decision, iterations):

    params = model.params
    mode = model.exog_info_fn_name

    lamb = params['Lambda']
    degrees = np.asarray([i**-1 for i in params['Degree']])
    degrees_mat = sparse.diags(degrees, format='csr')
    lamb_mod = degrees_mat.dot(lamb)

    dim = model.initial_state['Size']

    gradient_estimate = np.zeros((dim,dim))
    value_estimate = 0

    for i in range(iterations):
        if mode == 'Uniform':
            removals = uniform(model=model,
                               decision=decision)
        else:
            removals = weighted(model=model,
                                decision=decision)

        temp = np.zeros((dim,dim))
        for u, v in decision:
            temp[v][u] = 1
        for u, v in removals:
            temp[v][u] = -1
        X = sparse.csr_matrix(temp)
        gradient = fixed_gradient(model=model,
                                  lamb_mod=lamb_mod,
                                  X=X)
        gradient_estimate = gradient_estimate+gradient
        value = evaluate_function(model=model, X=X)
        value_estimate = value_estimate+value

    gradient_estimate = gradient_estimate/iterations
    value_estimate = value_estimate/iterations

    return gradient_estimate, value_estimate

# Computes exact gradient for a given set of edge
# additions and removals
def fixed_gradient(model, lamb_mod, X):

    state = model.state
    targets = model.initial_state['Targets']
    params = model.params

    dim = model.initial_state['Size']

    ones = np.zeros((dim,1))
    for i in range(len(targets)):
        ones[i,0] = 1
    state_level = np.reshape(state['Nodes'],newshape=(-1,1))

    W = nx.adjacency_matrix(state['Network'], weight=None).transpose()
    temp = normalize(W + X, norm='l1', axis=1)
    identity = sparse.identity(dim, format='csr').todense()
    phi = params['Lambda'].dot(temp) + identity - params['Lambda']

    mem = {}
    mem[0] = identity
    mem[1] = phi
    for i in range(2,params['T']):
        mem[i] = phi.dot(mem[i-1])

    result = np.zeros((dim, dim))
    for i in range(params['T']):
        temp = mem[i].T.dot(ones)
        temp = temp.dot(state_level.T)
        temp = temp.dot(mem[params['T']-1-i].T)
        result = result + temp

    return lamb_mod.dot(result)

# Executes the linear optimization step of the
# Gradient Based Policy
def linear_oracle(model, gradient):

    state = model.state
    targets = model.initial_state['Targets']
    params = model.params
    mode = model.exog_info_fn_name

    threshold = params['Alpha']*params['Beta']

    mem_struct = {}
    data_struct = {}
    decision_set = []

    for v in targets:
        data_struct[v] = {}

    for (u,v) in state['Decision Set']:

        const = mem_struct.get(v, -1)
        if const == -1:

            edges = state['Network'].in_edges([v], data = True)
            const = 0
            for (w, x, d) in edges:
                if mode == 'Uniform':
                    prob = 1/params['Degree'][v]
                else:
                    prob = (1-d['weight'])/(params['Degree'][v]-1)
                const = const+prob*gradient[x,w]

        mem_struct[v] = const

        score = gradient[v,u] - const
        if (score > threshold):
            data_struct[v][(u, v)] = score

    for key in data_struct:
        counter = Counter(data_struct[key])
        high = counter.most_common(params['Degree'][key])
        for i in high:
            decision_set.append(i[0])

    return decision_set

# Executes the Gradient Based Policy at time 0
def optimized_gradient_wrapper_policy(model):

    if model.state['Step'] == 0:
        return optimized_gradient_policy(model)
    else:
        return []
