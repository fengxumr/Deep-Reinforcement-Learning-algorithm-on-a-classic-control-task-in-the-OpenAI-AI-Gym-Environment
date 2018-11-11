import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

BATCH_SIZE = 30
BUFFER_SIZE = 10000
buffer = list()

# TODO: HyperParameters
GAMMA =  0.9                # discount factor
INITIAL_EPSILON =  0.5      # starting value of epsilon
FINAL_EPSILON =  0.01       # final value of epsilon
EPSILON_DECAY_STEPS =  400  # decay period
HIDDEN_DIM = 70

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
# W1 = tf.Variable(tf.truncated_normal([STATE_DIM, HIDDEN_DIM]))
# b1 = tf.Variable(tf.constant(0.01, shape = [HIDDEN_DIM]))
# W2 = tf.Variable(tf.truncated_normal([HIDDEN_DIM, ACTION_DIM]))
# b2 = tf.Variable(tf.constant(0.01, shape = [ACTION_DIM]))
# h_layer = tf.nn.relu(tf.matmul(state_in, W1) + b1)
h_layer = tf.layers.dense(state_in, HIDDEN_DIM, activation=tf.nn.relu)

# TODO: Network outputs
# q_values = tf.matmul(h_layer, W2) + b2
q_values = tf.layers.dense(h_layer, ACTION_DIM)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in),reduction_indices = 1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


saver = tf.train.Saver()
# checkpoint = tf.train.get_checkpoint_state("saved_networks")
# if checkpoint and checkpoint.model_checkpoint_path:
#     saver.restore(session, checkpoint.model_checkpoint_path)
#     print("Successfully loaded:", checkpoint.model_checkpoint_path)
# else:
#     print("Could not find old network weights")


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    if epsilon > FINAL_EPSILON:
        epsilon -= epsilon / EPSILON_DECAY_STEPS
    else:
        epsilon = FINAL_EPSILON

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        if len(buffer) > BUFFER_SIZE:
            buffer[random.randint(0, BUFFER_SIZE-1)] = (state, action, reward, next_state, done)
        else:
            buffer.append((state, action, reward, next_state, done))


        if len(buffer) > BATCH_SIZE:
            minibatch = random.sample(buffer, BATCH_SIZE)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            target = []
            Q_value_batch = q_values.eval(feed_dict={
                state_in:next_state_batch
                })
            for i in range(0,BATCH_SIZE):
                if minibatch[i][4]:
                    target.append(reward_batch[i])
                else :
                    target.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

            session.run([optimizer],feed_dict={
                target_in: target,
                action_in: action_batch,
                state_in: state_batch
                })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)
        saver.save(session, 'saved_networks/' + 'network' + '-dqn', global_step = episode)
        # if ave_reward >= 200:
        #     break

env.close()
