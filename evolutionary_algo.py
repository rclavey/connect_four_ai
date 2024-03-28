import numpy as np
import q_learn
import matplotlib.pyplot as plt
import csv
import os
import pickle
import multiprocessing
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

benchmark = {
    'win_rate': 0.95,  # The win rate threshold to become a new benchmark
    'agent': None,  # Initially, no benchmark agent
    'level': 0  # The initial level of benchmark (0 indicates the random agent is the benchmark)
}

def main(load=False):
    global benchmark
    # define the hyperparameters to optimize
    hyperparameters = ['temperature', 'cooling_rate', 'batch_size', 'alpha', 'beta', 'steps_per_update', 'learning_rate']

    # define the ranges for each hyperparameter
    ranges = {
        'temperature': (0.5, 2.0),
        'cooling_rate': (0.95, 1.0),
        'batch_size': (500, 2000),
        'alpha': (0.5, 1.0),
        'beta': (0.2, 0.6),
        'steps_per_update': (1, 20),
        'learning_rate': (0.0001, 0.01)
    }

    # define the population size and number of generations
    population_size = 14
    num_generations = 1000

    # define the number of episodes each individual will train for
    episodes = 1250
    # define the number of episodes to increase by if stagnant
    episode_step = 250


    # defines the margin of improvement considered significant
    improvement_margin = 0.01  
    # define the amount of generations without improvement until episode increase
    threshold_for_ep_increase = 5

    # define the mutation rate
    mutation_rate = 0.15

    # initialize best ai
    best_ai = None

    # initialize the population
    population = []

    # initialize score lists
    best_scores = []
    average_scores = []
    worst_scores = []

    errors_with_fitness = 0

    filenames = 'benchmark'
    # define the save file path
    save_file = f'{filenames}.pkl'


    # check if the save file exists
    if os.path.exists(save_file) and load == True:
        with open(save_file, 'rb') as f:
            data = pickle.load(f)
        population, best_scores, average_scores, worst_scores, generation, loaded_benchmark = data  
        # Load the benchmark agent's model if it exists
        if loaded_benchmark['agent'] is not None:
            with open(save_file, 'rb') as f:
                data = pickle.load(f)
            benchmark['agent'] = load_model(benchmark['agent'])


        benchmark.update(loaded_benchmark) 
    else:
        # initialize the population and score lists
        population = []
        best_scores = []
        average_scores = []
        worst_scores = []
        generation = 0

        for i in range(population_size):
            individual = {hp: np.random.uniform(*ranges[hp]) for hp in hyperparameters}
            population.append(individual)




    # open a csv file
    csvfile = open(f'{filenames}.csv', 'w', newline='')
    fieldnames = ['Generation', 'Episodes'] + ['fitness'] + hyperparameters
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


    # run the evolutionary algorithm
    for generation in range(generation, num_generations):
        print(f"Starting Generation {generation + 1}")

        # Prepare parameters for parallel fitness evaluation
        individual_params_list = [(individual, episodes, i) for i, individual in enumerate(population)]
        
        # Initialize multiprocessing Pool and obtain fitness results
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            fitness_results = pool.starmap(evaluate_fitness, [(ind_params, benchmark['level']) for ind_params in individual_params_list])
        
        # Update the population with the new fitness values and extract them for analysis
        for result in fitness_results:
            index = result['index']
            population[index]['fitness'] = result['fitness']

        # Now, extract just the fitness values for statistical analysis
        fitnesses = [ind['fitness'] for ind in population]

        # You can now safely calculate best, average, and worst scores
        best_score = max(fitnesses)
        average_score = sum(fitnesses) / len(fitnesses)
        worst_score = min(fitnesses)

        # After obtaining fitness_results from the pool.map(evaluate_fitness, individual_params_list)

        # Example logic to find the new benchmark based on returned fitness_results
        new_benchmark_model_path = None
        for result in fitness_results:
            if result.get('is_potential_benchmark'):
                # Assuming 'is_potential_benchmark' is a flag you set in evaluate_fitness
                # And 'model_path' is where you saved the model, if it was a potential benchmark
                new_benchmark_model_path = result['model_path']
                break  

        if new_benchmark_model_path:
            # Load the new benchmark model
            loaded_model = tf.keras.models.load_model(new_benchmark_model_path, compile=False, safe_mode=False)
            benchmark['agent'] = loaded_model
            benchmark['level'] += 1
            # Optionally, delete the temporary model file if you're saving a new one each time
            os.remove(new_benchmark_model_path)



        # Append the scores to the respective lists
        best_scores.append(best_score)
        average_scores.append(average_score)
        worst_scores.append(worst_score)


        # if the best score hasn't improved significantly for a certain number of generations, increase the number of episodes
        if generation >= threshold_for_ep_increase and best_score - best_scores[-threshold_for_ep_increase] < improvement_margin:
            episodes += episode_step  # increase amount of episodes by episode step
            mutation_rate *= 1.05
            print(f"Increasing number of episodes to {episodes} due to stagnant performance.")


        # select the best individuals to reproduce
        parents = select_parents(population)
        # plot the best, average, and worst scores
        plt.figure(figsize=(10, 5))
        plt.plot(best_scores, label='Best')
        plt.plot(average_scores, label='Average')
        plt.plot(worst_scores, label='Worst')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(f'{filenames}.png')

        # write best to csv file
        best_individual = select_best(population)
        csv_individual = best_individual.copy()  # create a copy of the dictionary
        csv_individual['Generation'] = generation + 1  # add the 'generation' key to the copy
        csv_individual['Episodes'] = episodes 
        csv_individual = {key: csv_individual[key] for key in csv_individual if key != 'q_network'}  # remove the 'q_network' key
        writer.writerow(csv_individual)
        csvfile.flush()
        os.fsync(csvfile.fileno())

                    
        # Generate offspring
        offspring = []
        # reduce pop size by one to add best individual back
        for _ in range(population_size - 1): 
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate, ranges)
            offspring.append(child)

        # Add the best individual from the previous generation to the offspring
        offspring.append(best_individual)
        # Replace the worst individuals with the new offspring
        population = replace_worst(population, offspring)

        # Define the model file path
        model_path = f"benchmark_model_{generation + 1}.keras"
        # If benchmark['agent'] is indeed a model instance, save it in the new recommended format.
        if isinstance(benchmark['agent'], tf.keras.Model):
            benchmark['agent'].save(model_path)  # Saves in the Keras format (.keras)
            benchmark['agent'] = model_path  # Updates the benchmark agent with the path

        # Prepare the data to be saved including the updated benchmark
        data = (population, best_scores, average_scores, worst_scores, generation + 1, benchmark)

        # Save the updated program state
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)



    csvfile.close()
    # Select the best individual from the final population
    best_individual = select_best(population)
    print(f"Best hyperparameters: {best_individual}")
    print(f"Errors with fitness: {errors_with_fitness}")
    print(f"The most recent benchmark ai was {benchmark['agent']} with a win rate threshold of {benchmark['win_rate']}, and a level of {benchmark['level']}.")


def choose_random_move(state):
    valid_actions = [col for col in range(len(state[0])) if state[0, col] == 0]
    return np.random.choice(valid_actions)


# create a fitness value for each algo
def evaluate_fitness(args, benchmark_level):
    global benchmark
    individual, episodes, index = args

    # Extract hyperparameters specific for the Q-network training
    steps_per_update = individual.get('steps_per_update', 1)
    learning_rate = individual.get('learning_rate', 0.001)
    temperature = individual.get('temperature', 1.0)
    cooling_rate = individual.get('cooling_rate', 0.99)
    batch_size = individual.get('batch_size', 32)
    alpha = individual.get('alpha', 0.6)  # Hyperparameter for the prioritized replay buffer
    beta = individual.get('beta', 0.4)  # Hyperparameter for the importance-sampling weights for the prioritized replay

    # Initialize and train the Q-network for this individual
    q_network = q_learn.main(episodes, temperature, cooling_rate, batch_size, alpha, beta, steps_per_update, learning_rate)

    # After training, we evaluate the performance of the trained Q-network against a random agent.
    num_games = 100
    wins = 0
    for game_index in range(num_games):
        game = q_learn.ConnectFour()
        state = game.get_state()
        game_over = False
        first_player = 1 if game_index % 2 == 0 else 2
        
        while not game.is_over() and not game_over:
            for player in [first_player, 3 - first_player]:
                if player == 1:
                    action = q_learn.choose_best_move_model(q_network, state)
                elif benchmark['agent'] is not None and player == 2:
                    action = choose_best_move_model(benchmark['agent'], state)  # Assuming this function uses the benchmark agent
                else:
                    action = choose_random_move(state)
                state, reward, done = game.make_move(action)
                if game.has_won(player):
                    if player == 1:
                        wins += 1
                    game_over = True
                    break

    # Calculate the fitness value based on the number of wins against the random agent
    win_rate = wins / num_games
    is_potential_benchmark = win_rate >= benchmark['win_rate']

    # Add benchmark level to fitness calculation
    fitness = win_rate + benchmark_level
    print(f"The fitness of individual {individual} is {fitness} because its win rate is {win_rate} and the benchmark is at level {benchmark_level}.")
    # Initialize model_path as None for cases when it's not a potential benchmark
    model_path = None

    if is_potential_benchmark:
        # Ensure the directory exists or is created
        model_directory = 'models'
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        
        model_path = f'models/temporary_model_{index}.keras'
        q_network.save(model_path)

    # Return the fitness along with other relevant details
    return {
        'index': index,
        'fitness': fitness,
        'hyperparameters': individual,
        'is_potential_benchmark': is_potential_benchmark,
        'model_path': model_path  # model_path will be None if not a potential benchmark
    }



# select the best half of individuals to be parents to next generation
def select_parents(population):
    num_parents = len(population) // 2
    parents = sorted(population, key=lambda ind: ind['fitness'], reverse=True)[:num_parents]
    return parents


# combine hyperparameters of two successful parents
def crossover(parent1, parent2):
    # single-point crossover
    crossover_point = np.random.randint(len(parent1))
    child = {}
    for i, hp in enumerate(parent1):
        if hp == 'Generation':
            continue
        if i < crossover_point:
            child[hp] = parent1[hp]
        else:
            child[hp] = parent2[hp]
    return child


# mutate the hyperparameters
def mutate(individual, mutation_rate, ranges):
    for hp in individual:
        if hp == 'fitness' or hp == 'q_network':
            continue
        if np.random.random() < mutation_rate:
            # define the proportion of the current value to use for the standard deviation
            std_dev_proportion = 0.1  
            
            # calculate the standard deviation based on the current value
            std_dev = std_dev_proportion * float(individual[hp])
            
            # add a normally distributed random number to the hyperparameter value
            mutated_value = individual[hp] + np.random.normal(0, std_dev)
            
            # ensure the mutated value is not negative
            mutated_value = max(0, mutated_value)
            
            # check if the mutated value is within the valid range
            min_value, max_value = ranges[hp]
            mutated_value = max(min_value, min(max_value, mutated_value))
            
            # update the hyperparameter value
            individual[hp] = mutated_value


# kill the worst algos and replace w/ offspring
def replace_worst(population, offspring):
    num_offspring = len(offspring)
    population = sorted(population, key=lambda ind: ind['fitness'], reverse=True)[:-num_offspring]
    population.extend(offspring)
    return population


# return best individual in the population
def select_best(population):
    return max(population, key=lambda ind: ind['fitness'])


def apply_action(state, action):
    game = q_learn.ConnectFour()
    game.board = state
    next_state, reward, done = game.make_move(action)
    return next_state


def get_winner(state):
    game = q_learn.ConnectFour()
    game.board = state
    if game.has_won(1):
        return 1
    elif game.has_won(2):
        return 2
    else:
        return 0


if __name__ == '__main__':
    main(True)
