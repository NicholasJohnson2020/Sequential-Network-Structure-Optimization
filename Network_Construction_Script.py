import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import networkx as nx

#load data
data = pd.read_pickle('processed_data.csv')

# Province dictionary to index dataset
prov_dict ={'Newfoundland and Labrador': 10,
            'Prince Edward Island': 11,
            'Nova Scotia': 12,
            'New Brunswick': 13,
            'Quebec': 24,
            'Ontario': 35,
            'Manitoba': 46,
            'Saskatchewan': 47,
            'Alberta': 48,
            'British Columbia': 59,
            'Yukon': 60,
            'Northwest Territories': 61,
            'Nunavut': 62}

# Quebec region dictionary to index dataset
quebec_region_dict = {'Bas-Saint-Laurent': 24901,
                      'Saguenay - Lac-Saint-Jean': 24902,
                      'Capitale-Nationale': 24903,
                      'Mauricie et du Centre-du-Québec': 24904,
                      'L\'Estrie': 24905,
                      'Région de Montréal': 24906,
                      'L\'Outaouais': 24907,
                      'L\'Abitibi-Témiscamingue': 24908,
                      'Côte-Nord': 24909,
                      'Gaspésie - Îles-de-la-Madeleine': 24911,
                      'Chaudière-Appalaches': 24912,
                      'Région de Laval': 24913,
                      'Région de Lanaudière': 24914,
                      'Région des Laurentides': 24915,
                      'Montérégie': 24916}

# Gender dictionary to index dataset
gender_dict = {'Male': 1.0, 'Female': 2.0}

# Age dictionary to index dataset
age_dict = {'12-14': 1.0, '15-17': 2.0, '18-19': 3.0, '20-24': 4.0,
            '25-29': 5.0, '30-34': 6.0, '35-39': 7.0, '40-44': 8.0,
            '45-49': 9.0, '50-54': 10.0, '55-59': 11.0, '60-64': 12.0,
            '65-69': 13.0, '70-74': 14.0, '75-79': 15.0, '80+': 16.0}

# Race dictionary to index dataset
race_dict = {'white': 1,
             'non-white': 2}

# Student status dictionary to index dataset
school_dict = {'yes': 1,
               'no': 2}

# Employment status dictionary to index dataset
employment_dict = {'yes': 1,
                   'no': 2,
                   'N/A': 6}

# Calculate nation wide [nonoverweight, overweight, obese] distribution

size = data.shape[0]

raw_nums = data['BMI (adjusted)'].values
nonover = data[np.less(raw_nums, 25)].shape[0]/size
over = data[np.logical_and(np.greater_equal(raw_nums, 25),
                           np.less(raw_nums, 30))].shape[0]/size
obese = data[np.greater_equal(raw_nums, 30)].shape[0]/size
    
state_dist = [nonover, over, obese]

# Create one hot encoders for the various features
encoders = {}

encoders['Gender'] = OneHotEncoder().fit(
    np.asarray(list(gender_dict)).reshape(-1, 1))
encoders['Age'] = OneHotEncoder().fit(
    np.asarray(list(age_dict)).reshape(-1, 1))
encoders['Race'] = OneHotEncoder().fit(
    np.asarray(list(race_dict)).reshape(-1, 1))
encoders['School'] = OneHotEncoder().fit(
    np.asarray(list(school_dict)).reshape(-1, 1))
encoders['Employment'] = OneHotEncoder().fit(
    np.asarray(list(employment_dict)).reshape(-1, 1))

# Person class represents a person's demographic attributes and health status
class Person:

    # Create a new person by sampling features from the
    # appropriate distributions
    def __init__(self, gender_d, age_d, race_d, school_d, employ_d):
        self.gender = np.random.choice(list(gender_dict), p = gender_d)
        self.age = np.random.choice(list(age_dict), p = age_d)
        self.race = np.random.choice(list(race_dict), p = race_d)

        self.school = np.random.choice(list(school_dict),
                                       p = school_d[self.age])
        self.employ = np.random.choice(list(employment_dict),
                                       p = employ_d[self.age])
        self.state = np.random.choice(['Non-overweight', 'Overweight', 'Obese'],
                                      p = state_dist)
        
        # Transform categorical features into one hot vector representations
        self.gender = encoders['Gender'].transform(
            np.asarray(self.gender).reshape(-1,1)).toarray()
        self.age = encoders['Age'].transform(
            np.asarray(self.age).reshape(-1,1)).toarray()
        self.race = encoders['Race'].transform(
            np.asarray(self.race).reshape(-1,1)).toarray()
        self.school = encoders['School'].transform(
            np.asarray(self.school).reshape(-1,1)).toarray()
        self.employ = encoders['Employment'].transform(
            np.asarray(self.employ).reshape(-1,1)).toarray()
    
    # Print representation of this person using
    # the explicit categroical features
    def print_features(self):
        print("Gender: ")
        print(encoders['Gender'].inverse_transform(self.gender))
        print("Age: ")
        print(encoders['Age'].inverse_transform(self.age))
        print("Race: ")
        print(encoders['Race'].inverse_transform(self.race))
        print("School: ")
        print(encoders['School'].inverse_transform(self.school))
        print("Employment: ")
        print(encoders['Employment'].inverse_transform(self.employ))
        print("Health: ")
        print(self.state)
    
    # Print representation of this person using implicit
    # one hot vector encodings of categorical features
    def print_encodings(self):
        
        print("Gender: ")
        print(self.gender)
        print("Age: ")
        print(self.age)
        print("Race: ")
        print(self.race)
        print("School: ")
        print(self.school)
        print("Employment: ")
        print(self.employ)
        
    # Returns the L2 norm of the difference of the feature vectors 
    # of node_a and node_b, which each represent an object of type Person
    def get_difference(node_a, node_b):
        
        gender_diff = np.sum(np.power(
            node_a.gender[0] - node_b.gender[0], 2))
        age_diff = np.sum(np.power(
            node_a.age[0] - node_b.age[0], 2))
        race_diff = np.sum(np.power(
            node_a.race[0] - node_b.race[0], 2))
        school_diff = np.sum(np.power(
            node_a.school[0] - node_b.school[0], 2))
        employ_diff = np.sum(np.power(
            node_a.employ[0] - node_b.employ[0], 2))
        return np.sqrt(gender_diff +
                       age_diff +
                       race_diff +
                       school_diff +
                       employ_diff)

# State dictionary that gives float value initialization for each health status
state_dict = {'Non-overweight': 1,
              'Overweight': 0.1,
              'Obese': 0}

# Generate n agents. For each agent, sample gender, age and race
# independently from the region specific distribution. Then,
# sample school status and employment status conditioned on age
# following the region specific distribution. Lastly, create the
# target set of obese individuals and the set of healthy
# individuals who have opted in.
def create_agents(region, num_agents, target_frac = 0.5, healthy_frac = 0.5):

    # Extract relevant data
    rel_data = data[data['Health Region'] == quebec_region_dict[region]]
    n = rel_data.shape[0]

    # Construct region specific distributions
    gender_dist = [rel_data[rel_data['Gender'] == gender_dict[i]].shape[0]/n 
                   for i in gender_dict.keys()]
    age_dist = [rel_data[rel_data['Age'] == age_dict[i]].shape[0]/n
                for i in age_dict.keys()]
    race_dist = [rel_data[rel_data['Cultural/Racial Background'] == 
                          race_dict[i]].shape[0]/n 
                 for i in race_dict.keys()]
    race_dist = race_dist/np.sum(race_dist)

    school_dists = {}
    employ_dists = {}
    for key in age_dict.keys():
        temp_data = rel_data[rel_data['Age'] == age_dict[key]]
        n_age = temp_data.shape[0]
        if n_age == 0:
            school_dists[key] = np.zeros(2)
            employ_dists[key] = np.zeros(3)
        else:
            school_dists[key] = [temp_data[temp_data['Attending School?'] == 
                                           school_dict[i]].shape[0]/n_age
                                 for i in school_dict.keys()]
            school_dists[key] = school_dists[key]/np.sum(school_dists[key])
            employ_dists[key] = [temp_data[temp_data['Employment Status'] ==
                                           employment_dict[i]].shape[0]/n_age
                                 for i in employment_dict.keys()]
            employ_dists[key] = employ_dists[key]/np.sum(employ_dists[key])
            
    # Sample agents
    agents = []
    obese_indices = []
    healthy_indices = []
    for i in range(num_agents):
        agents.append(Person(gender_dist,
                             age_dist,
                             race_dist,
                             school_dists,
                             employ_dists))
        if agents[i].state == 'Obese':
            obese_indices.append(i)
        else:
            healthy_indices.append(i)
    
    # Create the target subsets of the populations
    n_obese = int(np.ceil(target_frac*len(obese_indices)))
    obese_indices = np.random.choice(obese_indices,
                                     size = n_obese, replace = False)
    n_healthy = int(np.ceil(healthy_frac*len(healthy_indices)))
    healthy_indices = np.random.choice(healthy_indices,
                                       size = n_healthy, replace = False)
    
    return agents, obese_indices, healthy_indices

# Create a graph sampled using the region specific distribution and
# the Barabasi-Albert graph generation method with spatial
# preferential attachment
def create_graph(region, num_agents, m = 3, m_0 = 5, rho = 0.1):
    
    # Create the agents
    agents, obese_indices, healthy_indices = create_agents(
        region = region, num_agents = num_agents)
    
    # Initialize the graph with m_0 fully connected nodes
    G = nx.DiGraph()
    for i in range(m_0):
        G.add_node(i, state = state_dict[agents[i].state])
        for j in range(i):
            G.add_edge(i, j, weight = np.random.uniform(0,1))
            G.add_edge(j, i, weight = np.random.uniform(0,1))
        
    # Add remianing nodes using the Barabasi-Albert model
    # with spatial preferential attachment
    for i in range(m_0, num_agents):
        G.add_node(i, state = state_dict[agents[i].state])
        prob = []
        for j in range(i):
            dist = Person.get_difference(agents[i], agents[j])
            degree = G.in_degree(j)
            prob.append(np.exp(-rho*dist)*degree)
        prob = prob/np.sum(prob)
        new_edges = np.random.choice(range(i),
                                     size = m, replace = False, p = prob)
        for k in new_edges:
            G.add_edge(i, k, weight = np.random.uniform(0,1))
            G.add_edge(k, i, weight = np.random.uniform(0,1))
            
    # Normalize edge weights
    for node in G.nodes():
        weight_total = 0
        in_edges = G.in_edges([node], data = True)
        for u, v, d in in_edges:
            weight_total += d['weight']
        for u, v, d in in_edges:
            d['weight'] = d['weight']/weight_total
    
    return G, agents, obese_indices, healthy_indices

