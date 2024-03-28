import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Add, Lambda, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras import backend as k
import os
from glob import glob


# create a prioritized replay buffer class to improve performance
class PrioritizedReplayBuffer:
    # initialize the replay buffer to accept maximum size of the buffer and the alpha value
    # the alpha value determines probabilities such that 0 is uniform selection and 1 is by importance
    def __init__(self, max_size, alpha):
        self.buffer = []
        self.priorities = []
        self.max_size = max_size
        self.index = 0
        self.alpha = alpha

    # create add method and accept experience and td error as parameters
    def add(self, experience, td_error):
        # set priority equal to the absolute value of the td error, plus small number to the power of alpha hyperparameter
        priority = (abs(td_error) + 1e-5) ** self.alpha
        # if the buffer is not full, add experience to it
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        # if the buffer is full, replace the least prioritized buffer
        else:
            self.buffer[self.index] = experience
            self.priorities[self.index] = priority
        self.index = (self.index + 1) % self.max_size

    # create a sample method with batch size and beta as parameters 
    # beta value determines how much correction is done
    def sample(self, batch_size, beta):
        # set priorities as a numpy array
        priorities = np.array(self.priorities)
        # set probabilities as a ratio of priority to total
        probabilities = priorities / np.sum(priorities)
        # set indices to a random value with varying probability
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        # set experiences to the buffer for each of the indices
        experiences = [self.buffer[i] for i in indices]
        # set weight ewual to the length of buffer times probability of specific index raised to the negative power of beta
        weights = (len(self.buffer) * probabilities[indices]) ** -beta
        # set weights equal to itself divided by max value of itself
        weights /= weights.max()
        # return values
        return experiences, indices, weights

    # create a method to update prioties with indices and td error as parameters
    def update_priorities(self, indices, td_errors):
        # update each priority
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority


# Custom optimizer for gradient accumulation
# accumulate gradients over multiple steps to save on computation because original version crashes after about 1000 episodes
class GradientAccumulation(tf.keras.optimizers.Adam):
    # intialize the parant class with steps per update, arguments and keyword arguments
    def __init__(self, steps_per_update, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_update = steps_per_update
        self.gradient_accumulators = None

    # define method to create gradient accumulators for each variable in list
    def _create_slots(self, var_list):
        super()._create_slots(var_list)
        self.gradient_accumulators = [self.add_slot(var, 'accumulator') for var in var_list]

    # define method to take gradient and variables as inputs
    def apply_gradients(self, grads_and_vars, name=None):
        grads, vars = list(zip(*grads_and_vars))
        
        # check to see if accumulators are initialized and ensure they have same shape as gradients
        if self.gradient_accumulators is None:
            self.gradient_accumulators = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in grads]

        # define a function to return variables after gradient is applied
        def apply_grads():
            return super(GradientAccumulation, self).apply_gradients(zip(grads, vars), name)

        # Accumulate gradients over multiple steps
        for grad, accumulator in zip(grads, self.gradient_accumulators):
            accumulator.assign_add(grad)

        # Apply gradients every `steps_per_update` steps
        condition = tf.equal(self.iterations % self.steps_per_update, 0)
        tf.cond(condition, apply_grads, lambda: tf.constant(0, dtype=tf.int64))

        # Reset gradient accumulators after gradients are applied
        with tf.control_dependencies([apply_grads()]):
            for accumulator in self.gradient_accumulators:
                accumulator.assign(tf.zeros_like(accumulator))


    # override the default "__setattr__" method to ensure gradient accumulators are properly initialized
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "_create_slots_was_called":
            self._create_slots(self.variables())

    # add methods to be able to load saved file in evolutionary_ai.py file
    def get_config(self):
        config = super(GradientAccumulation, self).get_config()
        config.update({
            'steps_per_update': self.steps_per_update,
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# create the layers for the q learning network
def create_q_network(steps_per_update, learning_rate):
    # create architecture for dueling dqn
    input_layer = Input(shape=(6, 7))
    flatten_layer = Flatten()(input_layer)
    
    # Common hidden layers with 128, 64, 32 neurons
    dense1 = Dense(256, activation='gelu')(flatten_layer)
    dense2 = Dense(128, activation='gelu')(dense1)
    dense3 = Dense(64, activation='gelu')(dense2)
    
    # State value stream (get q-value for state of board)
    state_value_dense = Dense(16, activation='gelu')(dense3)
    state_value = Dense(1, activation='linear')(state_value_dense)
    
    # Action advantage stream (get q-value for potential actions)
    action_advantage_dense = Dense(16, activation='gelu')(dense3)
    action_advantage = Dense(7, activation='linear')(action_advantage_dense)
    
    # Combine state value and action advantage streams to get final Q-values
    q_values = Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True), output_shape=(7,))(action_advantage)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=q_values)
    
    # Create a gradient accumulation optimizer
    optimizer = GradientAccumulation(steps_per_update, learning_rate)
    model.compile(loss='mse', optimizer=optimizer)

    return model


# create target q-network for double dqn infastructure
def create_target_q_network(q_network, steps_per_update, learning_rate):
    target_q_network = create_q_network(steps_per_update, learning_rate)
    update_target_q_network(target_q_network, q_network)
    return target_q_network


# update double dqn
def update_target_q_network(target_q_network, q_network):
    target_q_network.set_weights(q_network.get_weights())


# use a softmax function for action selection (greedy epsilon did not work as well as softmax)
def softmax(x):
    # Shift the values of x so that the maximum value in the array is 0 (no more NAN errors)
    x_shifted = x - np.max(x)
    
    # If any of the shifted values are very large negative numbers, set them to a large negative number
    x_shifted[x_shifted < -100] = -100
    
    # Exponentiate the shifted values and normalize
    e_x = np.exp(x_shifted)
    return e_x / e_x.sum()


# apply softmax
def get_softmax_action(q_values, temperature, valid_actions):
    # Clip the Q-values to prevent overflow/underflow
    q_values_clipped = np.clip(q_values[valid_actions] / temperature, -100, 100)

    # Apply softmax to the clipped Q-values
    softmax_probs = softmax(q_values_clipped)

    # Choose an action based on the softmax probabilities
    return np.random.choice(valid_actions, p=softmax_probs)


# make a function to update the q network
def update_q_network(q_network, target_q_network, states, actions, rewards, next_states, dones, weights):
    # set target next q values to the prediction based on target q network parameter and next states parameter
    target_next_q_values = target_q_network.predict(next_states,verbose=0)
    # set best actions to the max values in predictions
    best_actions = np.argmax(q_network.predict(next_states, verbose=0), axis=1)
    # set the target value to the reward plus the anticipated future rewards multiplied by gamma, the discount factor. If game is done, multiply by 0 to cancel.
    target_q_values = rewards + gamma * target_next_q_values[np.arange(len(rewards)), best_actions] * (1 - dones)
    # set predicted q values
    predicted_q_values = q_network.predict(states,verbose=0)
    # calculated td errors by subtracting predicted values from target values
    td_errors = target_q_values - predicted_q_values[np.arange(len(rewards)), actions]
    # update the priorites of the replay buffer
    replay_buffer.update_priorities(indices, td_errors)
    # replace the predicted q values with the target q values
    for i, action in enumerate(actions):
        predicted_q_values[i][action] = target_q_values[i]
    # Multiply the TD errors by the importance sampling weights
    td_errors *= weights
    q_network.fit(states, predicted_q_values, batch_size=batch_size, epochs=1, verbose=0)

    return td_errors

# create a connect four class to be able to train the ai
class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.current_player = 1
        self.last_move = None

    def get_state(self):
        return self.board.copy()

    def is_full(self, column):
        return self.board[0, column] != 0

    def make_move(self, column):
        if self.is_full(column):
            raise ValueError(f"Column {column} is full.")

        for row in reversed(range(6)):
            if self.board[row, column] == 0:
                self.board[row, column] = self.current_player
                break

        # add intermediate rewards to increase complexity of strategy developed
        alignment_reward = 0
        if self.check_win(3, self.current_player):
            alignment_reward += 10
        elif self.check_win(2, self.current_player):
            alignment_reward += 5

        # create negative reward when opponent player does something good
        opponent_player = 3 - self.current_player
        if self.check_win(3, opponent_player):
            alignment_reward -= 25
        elif self.check_win(2, opponent_player):
            alignment_reward -= 10

        # Check if the current player has won
        if self.has_won(self.current_player):
            reward = 100 + alignment_reward
        # check if opponent has won
        elif self.has_won(opponent_player):
            reward = -100
        else:
            reward = alignment_reward

        # Get the game over status before switching players
        game_over = self.is_over()

        # Switch player (1 to 2, or 2 to 1)
        self.current_player = 3 - self.current_player  

        return self.get_state(), reward, game_over

    def check_win(self, length, player):
        rows = len(self.board)
        cols = len(self.board[0])

        # Check for horizontal win
        for row in range(rows):
            for col in range(cols - length + 1):
                if all(self.board[row][col + i] == player for i in range(length)):
                    return True

        # Check for vertical win
        for row in range(rows - length + 1):
            for col in range(cols):
                if all(self.board[row + i][col] == player for i in range(length)):
                    return True

        # Check for ascending diagonals
        for row in range(length - 1, rows):
            for col in range(cols - length + 1):
                if all(self.board[row - i][col + i] == player for i in range(length)):
                    return True

        # Check for descending diagonals
        for row in range(rows - length + 1):
            for col in range(cols - length + 1):
                if all(self.board[row + i][col + i] == player for i in range(length)):
                    return True

        return False

    def has_won(self, player):
        return self.check_win(4, player)

    def is_over(self):
        return self.has_won(1) or self.has_won(2) or np.all(self.board != 0)


def get_best_action(model, board_state):
    # Get Q-values for each action by predicting with the model
    q_values = model.predict(board_state[np.newaxis])[0]

    # Get the indices of the valid actions (not full columns)
    valid_actions = [col for col in range(7) if board_state[0, col] == 0]

    # Choose the action with the highest Q-value among the valid actions
    best_action = valid_actions[np.argmax(q_values[valid_actions])]

    return best_action

def choose_random_move(state):
    valid_actions = [col for col in range(7) if state[0, col] == 0]
    return np.random.choice(valid_actions)

def main(num_episodes, temperature, cooling_rate, batch_size, alpha, beta, steps_per_update, learning_rate, file=None ):
    # create a q network with variable
    q_network = create_q_network(steps_per_update, learning_rate)
    # create target q network
    target_q_network = create_target_q_network(q_network, steps_per_update, learning_rate)

    # possibly move
    initial_priority = 1.0
    replay_buffer = PrioritizedReplayBuffer(5_000, alpha)

    if file is not None:
        q_network = load_trained_model(file)
    else:
        q_network = create_q_network(steps_per_update, learning_rate)



    # iterate through the amount of episodes
    for episode in range(num_episodes):
        print(f'Episode: {episode} / {num_episodes} \n {100 * episode/num_episodes} % done training')
        # in each episode, create a game of connect four
        game = ConnectFour()
        state = game.get_state()
        done = False

        if episode % 250 == 0:
            update_target_q_network(target_q_network, q_network)

        # iterate until game is over
        while not done:
            # define the valid actions as a list of moves the agent can take (all columns that arent full)
            valid_actions = [col for col in range(7) if not game.is_full(col)]
            # define the q-values by passing through the q_network
            q_values = q_network.predict(state[np.newaxis],verbose=0)
            # define action based on predicted q_value and epsilon greedy
            action = get_softmax_action(q_values[0], temperature, valid_actions)
            # return the game state and reward of the game, as well as if the game is done or not after action is taken
            next_state, reward, done = game.make_move(action)
            # update the q_network using the update function and the variables defined above using experience replay buffer
            if len(replay_buffer.buffer) >= batch_size:
                experiences, indices, weights = replay_buffer.sample(batch_size, beta)
                states, actions, rewards, next_states, dones = zip(*experiences)
                states = np.array(states)
                experience = (state, action, reward, next_state, done)
                replay_buffer.add(experience, initial_priority)  
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                td_errors = update_q_network(q_network, target_q_network, states, actions, rewards, next_states, dones, weights)
                for i, index in enumerate(indices):
                    replay_buffer.update_priority(index, td_errors[i])
            # proceed to next state of the game
            state = next_state
        # delete the game instance to use less memory
        del game
        # clean any unused resources 
        k.clear_session()

        # Save the q_network weights every 500 episodes


        # Define the folder for save files
        save_folder = 'periodic_save_files'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Function to save model weights with PID and episode information
        def save_model_weights(q_network, episode):
            pid = os.getpid()  # Gets the current process ID
            # Define the save file name with PID and episode
            unique_filename = f'periodic_save_{pid}_{episode}.weights.h5'
            save_path = os.path.join(save_folder, unique_filename)
            q_network.save_weights(save_path)
            print(f'Weights have been saved to {save_path}')
            cleanup_save_files()

        # Optional: Function to delete older save files
        def cleanup_save_files(keep_last_n=28):
            save_folder = 'periodic_save_files'
            save_files = sorted(glob(os.path.join(save_folder, 'periodic_save_*.weights.h5')))
            files_to_delete = save_files[:-keep_last_n]
            for file_path in files_to_delete:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f'Deleted old save file: {file_path}')
                except FileNotFoundError:
                    print(f"File not found and couldn't be deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


        # Example usage within your training loop
        if episode % 2000 == 0:
            save_model_weights(q_network, episode)

        # decrease temperature value so that network will explore less and exploit more as more iterations take place
        temperature = max(0.01, temperature * cooling_rate)


    # return the q_network once iterations have finished
    return q_network


def load_trained_model(file):
    q_network = create_q_network()
    q_network.load_weights(file)
    return q_network

# choose best move (for context with connnect_four game)
def choose_best_move(positions, player, file):
    trained_q_network = load_trained_model(file)

    adjusted_state = np.array(positions, dtype=int)
    adjusted_state[positions == player] = 1
    adjusted_state[positions == (3 - player)] = -1

    q_values = trained_q_network.predict(adjusted_state[np.newaxis],verbose=0)[0]
    valid_actions = [col for col in range(len(positions[0])) if positions[0, col] == 0]
    return valid_actions[np.argmax(q_values[valid_actions])]

# choose best move with model already made (for use with tester)
def choose_best_move_model(model, state):
    # Get the Q-values for all actions
    q_values = model.predict(state[np.newaxis])[0]

    # Get the valid actions (columns that are not full)
    valid_actions = [col for col in range(7) if state[0, col] == 0]

    # Choose the action with the highest Q-value among the valid actions
    best_action = valid_actions[np.argmax(q_values[valid_actions])]

    return best_action


def list_moves(positions, player, file):
    trained_q_network = load_trained_model(file)

    adjusted_state = np.array(positions, dtype=int)
    adjusted_state[positions == player] = 1
    adjusted_state[positions == (3 - player)] = -1

    q_values = trained_q_network.predict(adjusted_state[np.newaxis],verbose=0)[0]
    valid_actions = [col for col in range(len(positions[0])) if positions[0, col] == 0]
    return q_values[valid_actions]


if __name__ == "__main__":
    
    file = 'alpha_high_p.h5'

    # use below to train model
    trained_q_network = main(2000, 1.6766059583349224, 0.9737425540223724, 1127.4794541257743, 0.6893362287549736, 0.2499451558403795, 1.5673060295703025, 0.008838368781126052)

    # Save the trained model weights to an HDF5 file
    trained_q_network.save_weights(file)

    print(f"Model weights saved to '{file}'")
