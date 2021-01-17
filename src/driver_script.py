# Import relevant packages
import numpy as np
import time
import sys
import pickle

# Import custom scripts
import network_construction_script as ncs
import snvironment_script as es
import policy_script as ps

global_start_time = time.time()

# Verify the number of command line arguments
assert len(sys.argv) == 4

# Function to execute a single simulation
def driver_script(initial_network,
                  target_nodes,
                  healthy_nodes,
                  lamda_vec,
                  exog_mode,
                  objective_mode,
                  policy,
                  T = 10,
                  alpha = 0.01,
                  beta = 1.0001,
                  mu = 0.1,
                  debug = False):

    start_time = time.time()

    # Call the environment constructor
    Model = es.ObesityModel(initial_network = initial_network,
                            target_nodes = target_nodes,
                            healthy_nodes = healthy_nodes,
                            lamda_vec = lamda_vec,
                            exog_info_fn_name = exog_mode,
                            objective_fn_name = objective_mode,
                            T=T,
                            alpha = alpha,
                            beta=beta,
                            mu = mu)

    decisions = []
    objective_val = []

    # Run the simulation
    for i in range(T):

        decision = policy(Model)
        decisions.append(decision)
        Model.step(decision=decision, debug=debug)
        objective_val.append(float(Model.objective))

    time_elapsed = time.time() - start_time
    final_state = [Model.state['Nodes'][i][0] for i in target_nodes]
    final_state = sum(final_state)

    return objective_val, decisions, time_elapsed, final_state

# Policy dictionary
policies = {'Control': ps.control_policy,
            'Perpetual Random': ps.random_policy,
            'Initial Random': ps.random_wrapper_policy,
            'Myopic': ps.optimized_myopic_policy,
            'One Step Lookahead': ps.optimized_simple_lookahead_wrapper_policy,
            'Modifed Reconnect': ps.optimized_gradient_wrapper_policy}

# Parameter values that will be explored
num_agents_vals = [100, 200, 400, 600, 800, 1000, 1500, 2000]
m_vals = [4, 6, 10, 15, 20, 30, 40, 50, 75]
m_0_vals = [5, 10, 15, 20, 30, 40, 50, 75]
T_vals = list(np.arange(10,101,10))

objective_name = ['Cumulative', 'Terminal']
random_name = ['Uniform', 'Weighted']

exp_results = {}
exp_results['Cumulative'] = {'Uniform': {},
                             'Weighted': {}}
exp_results['Terminal'] = {'Uniform': {},
                             'Weighted': {}}

trials = int(sys.argv[1])

if sys.argv[2] == 'num_agents':

    default_vals = {'num_agents': 1000,
                    'm': 15,
                    'm_0': 50,
                    'T': 20,
                    'mu': 0.1,
                    'beta': 1.001}

    for obj_name in objective_name:
        for exog_name in random_name:
            for policy in policies.keys():
                new_dict = {}
                for n in num_agents_vals:
                    temp_dict = {'Objectives': [],
                                'Times': [],
                                'Final States': []}
                    new_dict[n] = temp_dict
                exp_results[obj_name][exog_name][policy] = new_dict

    print('Executing experiments for the parameter: ' + sys.argv[2])

    for obj_name in objective_name:
        for exog_name in random_name:
            config_start = time.time()
            print('Current configuration is: '+obj_name+' x '+exog_name)
            current_dict = exp_results[obj_name][exog_name]
            for n in num_agents_vals:
                param_start = time.time()
                print('Executing trials for '+sys.argv[2]+' = '+str(n))
                for i in range(trials):
                    graph, _, obese_ind, healthy_ind = ncs.create_graph(
                            region = 'Région de Montréal',
                            num_agents=n,
                            m = default_vals['m'],
                            m_0 = default_vals['m_0'],
                            rho = 0.1)
                    lamb_vec = np.random.uniform(size=n)
                    for key in policies.keys():
                        obj, _, tim, final = driver_script(
                                initial_network=graph,
                                target_nodes=obese_ind,
                                healthy_nodes=healthy_ind,
                                lamda_vec=lamb_vec,
                                exog_mode=exog_name,
                                objective_mode=obj_name,
                                policy=policies[key],
                                T = default_vals['T'],
                                alpha = 0.0005,
                                beta = default_vals['beta'],
                                mu = default_vals['mu'],
                                debug = False)

                        current_dict[key][n]['Objectives'].append(
                                obj[default_vals['T']-1])
                        current_dict[key][n]['Times'].append(tim)
                        current_dict[key][n]['Final States'].append(final)

                print('Trials for '+sys.argv[
                        2]+' = '+str(n)+' took '+str(
                        time.time()-param_start))

            for key in policies.keys():
                for n in num_agents_vals:
                    current_dict[key][n]['Mean Objective'] = np.mean(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Std Objective'] = np.std(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Mean Time'] = np.mean(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Std Time'] = np.std(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Mean Final'] = np.mean(
                            current_dict[key][n]['Final States'])
                    current_dict[key][n]['Std Final'] = np.std(
                            current_dict[key][n]['Final States'])

            print('Configuration time: '+str(
                    time.time()-config_start))

    params_freeze = default_vals.copy()
    params_freeze['num_agents'] = num_agents_vals
    exp_results['Parameters'] = params_freeze

    with open(sys.argv[3], 'wb') as handle:
        pickle.dump(exp_results, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print('The total execution time is: ' + str(
            time.time()-global_start_time))

elif sys.argv[2] == 'm':

    default_vals = {'num_agents': 1000,
                    'm': 4,
                    'm_0': 75,
                    'T': 20,
                    'mu': 0.1,
                    'beta': 1.001}

    for obj_name in objective_name:
        for exog_name in random_name:
            for policy in policies.keys():
                new_dict = {}
                for n in m_vals:
                    temp_dict = {'Objectives': [],
                                'Times': [],
                                'Final States': []}
                    new_dict[n] = temp_dict
                exp_results[obj_name][exog_name][policy] = new_dict

    print('Executing experiments for the parameter: ' + sys.argv[2])

    for obj_name in objective_name:
        for exog_name in random_name:
            config_start = time.time()
            print('Current configuration is: '+obj_name+' x '+exog_name)
            current_dict = exp_results[obj_name][exog_name]
            for n in m_vals:
                param_start = time.time()
                print('Executing trials for '+sys.argv[2]+' = '+str(n))
                for i in range(trials):
                    graph, _, obese_ind, healthy_ind = ncs.create_graph(
                            region = 'Région de Montréal',
                            num_agents = default_vals['num_agents'],
                            m = n,
                            m_0 = default_vals['m_0'],
                            rho = 0.1)
                    lamb_vec = np.random.uniform(
                            size=default_vals['num_agents'])
                    for key in policies.keys():
                        obj, _, tim, final = driver_script(
                                initial_network=graph,
                                target_nodes=obese_ind,
                                healthy_nodes=healthy_ind,
                                lamda_vec=lamb_vec,
                                exog_mode=exog_name,
                                objective_mode=obj_name,
                                policy=policies[key],
                                T = default_vals['T'],
                                alpha = 0.0005,
                                beta = default_vals['beta'],
                                mu = default_vals['mu'],
                                debug = False)

                        current_dict[key][n]['Objectives'].append(
                                obj[default_vals['T']-1])
                        current_dict[key][n]['Times'].append(tim)
                        current_dict[key][n]['Final States'].append(final)

                print('Trials for '+sys.argv[
                        2]+' = '+str(n)+' took '+str(
                        time.time()-param_start))

            for key in policies.keys():
                for n in m_vals:
                    current_dict[key][n]['Mean Objective'] = np.mean(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Std Objective'] = np.std(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Mean Time'] = np.mean(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Std Time'] = np.std(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Mean Final'] = np.mean(
                            current_dict[key][n]['Final States'])
                    current_dict[key][n]['Std Final'] = np.std(
                            current_dict[key][n]['Final States'])

            print('Configuration time: ' + str(
                    time.time()-config_start))

    params_freeze = default_vals.copy()
    params_freeze['m'] = m_vals
    exp_results['Parameters'] = params_freeze

    with open(sys.argv[3], 'wb') as handle:
        pickle.dump(exp_results, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print('The total execution time is: ' + str(
            time.time()-global_start_time))

elif sys.argv[2] == 'm_0':

    default_vals = {'num_agents': 1000,
                    'm': 4,
                    'm_0': 10,
                    'T': 20,
                    'mu': 0.1,
                    'beta': 1.001}


    for obj_name in objective_name:
        for exog_name in random_name:
            for policy in policies.keys():
                new_dict = {}
                for n in m_0_vals:
                    temp_dict = {'Objectives': [],
                                'Times': [],
                                'Final States': []}
                    new_dict[n] = temp_dict
                exp_results[obj_name][exog_name][policy] = new_dict

    print('Executing experiments for the parameter: '+sys.argv[2])

    for obj_name in objective_name:
        for exog_name in random_name:
            config_start = time.time()
            print('Current configuration is: '+obj_name +' x '+exog_name)
            current_dict = exp_results[obj_name][exog_name]
            for n in m_0_vals:
                param_start = time.time()
                print('Executing trials for '+sys.argv[2]+' = '+str(n))
                for i in range(trials):
                    graph, _, obese_ind, healthy_ind = ncs.create_graph(
                            region = 'Région de Montréal',
                            num_agents = default_vals['num_agents'],
                            m = default_vals['m'],
                            m_0 = n,
                            rho = 0.1)
                    lamb_vec = np.random.uniform(
                            size=default_vals['num_agents'])
                    for key in policies.keys():
                        obj, _, tim, final = driver_script(
                                initial_network=graph,
                                target_nodes=obese_ind,
                                healthy_nodes=healthy_ind,
                                lamda_vec=lamb_vec,
                                exog_mode=exog_name,
                                objective_mode=obj_name,
                                policy=policies[key],
                                T = default_vals['T'],
                                alpha = 0.0005,
                                beta = default_vals['beta'],
                                mu = default_vals['mu'],
                                debug = False)

                        current_dict[key][n]['Objectives'].append(
                                obj[default_vals['T']-1])
                        current_dict[key][n]['Times'].append(tim)
                        current_dict[key][n]['Final States'].append(final)

                print('Trials for '+sys.argv[
                        2]+' = '+str(n)+' took '+str(
                        time.time()-param_start))

            for key in policies.keys():
                for n in m_0_vals:
                    current_dict[key][n]['Mean Objective'] = np.mean(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Std Objective'] = np.std(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Mean Time'] = np.mean(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Std Time'] = np.std(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Mean Final'] = np.mean(
                            current_dict[key][n]['Final States'])
                    current_dict[key][n]['Std Final'] = np.std(
                            current_dict[key][n]['Final States'])

            print('Configuration time: ' + str(
                    time.time()-config_start))

    params_freeze = default_vals.copy()
    params_freeze['m_0'] = m_0_vals
    exp_results['Parameters'] = params_freeze

    with open(sys.argv[3], 'wb') as handle:
        pickle.dump(exp_results, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print('The total execution time is: ' + str(
            time.time()-global_start_time))

elif sys.argv[2] == 'T':

    default_vals = {'num_agents': 1000,
                    'm': 15,
                    'm_0': 50,
                    'T': 20,
                    'mu': 0.1,
                    'beta': 1.001}


    for obj_name in objective_name:
        for exog_name in random_name:
            for policy in policies.keys():
                new_dict = {}
                for n in T_vals:
                    temp_dict = {'Objectives': [],
                                'Times': [],
                                'Final States': []}
                    new_dict[n] = temp_dict
                exp_results[obj_name][exog_name][policy] = new_dict

    print('Executing experiments for the parameter: '+sys.argv[2])

    for obj_name in objective_name:
        for exog_name in random_name:
            config_start = time.time()
            print('Current configuration is: '+obj_name+' x '+exog_name)
            current_dict = exp_results[obj_name][exog_name]
            graph, _, obese_ind, healthy_ind = ncs.create_graph(
                        region = 'Région de Montréal',
                        num_agents = default_vals['num_agents'],
                        m = default_vals['m'],
                        m_0 = default_vals['m_0'],
                        rho = 0.1)
            for n in T_vals:
                param_start = time.time()
                print('Executing trials for '+sys.argv[2]+' = '+str(n))
                for i in range(trials):
                    lamb_vec = np.random.uniform(
                            size=default_vals['num_agents'])
                    for key in policies.keys():
                        obj, _, tim, final = driver_script(
                                initial_network=graph,
                                target_nodes=obese_ind,
                                healthy_nodes=healthy_ind,
                                lamda_vec=lamb_vec,
                                exog_mode=exog_name,
                                objective_mode=obj_name,
                                policy=policies[key],
                                T = n,
                                alpha = 0.0005,
                                beta = default_vals['beta'],
                                mu = default_vals['mu'],
                                debug = False)

                        current_dict[key][n]['Objectives'].append(
                                obj[n-1])
                        current_dict[key][n]['Times'].append(tim)
                        current_dict[key][n][
                                'Final States'].append(final)

                print('Trials for '+sys.argv[
                        2]+' = '+str(n)+' took '+str(
                        time.time()-param_start))

            for key in policies.keys():
                for n in T_vals:
                    current_dict[key][n]['Mean Objective'] = np.mean(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Std Objective'] = np.std(
                            current_dict[key][n]['Objectives'])
                    current_dict[key][n]['Mean Time'] = np.mean(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Std Time'] = np.std(
                            current_dict[key][n]['Times'])
                    current_dict[key][n]['Mean Final'] = np.mean(
                            current_dict[key][n]['Final States'])
                    current_dict[key][n]['Std Final'] = np.std(
                            current_dict[key][n]['Final States'])

            print('Configuration time: ' + str(
                    time.time()-config_start))

    params_freeze = default_vals.copy()
    params_freeze['T'] = T_vals
    exp_results['Parameters'] = params_freeze

    with open(sys.argv[3], 'wb') as handle:
        pickle.dump(exp_results, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print('The total execution time is: ' + str(
            time.time()-global_start_time))

else:

    print('Invalid Second Argument Entered')
