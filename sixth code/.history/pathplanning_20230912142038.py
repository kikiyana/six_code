import tensorflow as tf
import numpy as np
import gym
import threading
import multiprocessing

# 创建全局参数
global_episode = 0
max_global_episodes = 10000
global_step = 0
global_step_lock = threading.Lock()
num_workers = multiprocessing.cpu_count()

# A3C 超参数
learning_rate = 0.001
gamma = 0.99
n_step = 5
state_dim = 4
action_dim = 2
hidden_dim = 128

# 创建 Actor 网络
class ActorNetwork(tf.keras.Model):
    def __init__(self):
        # super(ActorNetwork, self).__init__()
        self.fc1 = tf.layers.Dense(hidden_dim, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(action_dim, activation=tf.nn.softmax)

    def call(self, state):
        x = self.fc1(state)
        return self.fc2(x)

# 创建 Critic 网络
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = tf.layers.Dense(hidden_dim, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(1)

    def call(self, state):
        x = self.fc1(state)
        return self.fc2(x)

# 创建全局 Actor-Critic 网络
global_actor_network = ActorNetwork()
global_critic_network = CriticNetwork()

# 创建 A3C 工作线程
class Worker(threading.Thread):
    def __init__(self, worker_id, global_actor_network, global_critic_network):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.actor_network = ActorNetwork()
        self.critic_network = CriticNetwork()
        self.actor_network.build([None, state_dim])
        self.critic_network.build([None, state_dim])
        self.actor_network.set_weights(global_actor_network.get_weights())
        self.critic_network.set_weights(global_critic_network.get_weights())

    def run(self):
        global global_episode, global_step
        env = gym.make('CartPole-v1')
        while global_episode < max_global_episodes:
            state = env.reset()
            done = False
            episode_reward = 0
            episode_states, episode_actions, episode_values, episode_rewards = [], [], [], []

            while not done:
                state = state.reshape([1, state_dim])
                action_prob = self.actor_network(state)[0]
                action = np.random.choice(action_dim, p=action_prob)
                value = self.critic_network(state)[0]
                episode_states.append(state)
                episode_actions.append(action)
                episode_values.append(value)

                next_state, reward, done, _ = env.step(action)
                episode_rewards.append(reward)
                state = next_state
                episode_reward += reward
                global_step_lock.acquire()
                global_step += 1
                global_step_lock.release()

                if global_step % n_step == 0 or done:
                    if not done:
                        next_state = next_state.reshape([1, state_dim])
                        next_value = self.critic_network(next_state)[0]
                    else:
                        next_value = 0

                    discounted_rewards = []
                    advantage = 0
                    for r in episode_rewards[::-1]:
                        advantage = r + gamma * advantage
                        discounted_rewards.append(advantage)
                    discounted_rewards.reverse()

                    episode_states = np.vstack(episode_states)
                    episode_actions = np.array(episode_actions)
                    episode_values = np.array(episode_values)
                    discounted_rewards = np.array(discounted_rewards)

                    with global_step_lock:
                        global_episode += 1

                    with tf.Session() as sess:
                        policy_loss = -tf.reduce_sum(
                            tf.log(tf.reduce_sum(tf.multiply(
                                tf.one_hot(episode_actions, action_dim), self.actor_network(episode_states)), axis=1)))
                        policy_loss = tf.reduce_sum(policy_loss * discounted_rewards)
                        value_loss = tf.reduce_sum(tf.square(episode_values - discounted_rewards))
                        total_loss = policy_loss + value_loss

                        grads_actor = tf.gradients(total_loss, self.actor_network.trainable_variables)
                        grads_critic = tf.gradients(value_loss, self.critic_network.trainable_variables)

                        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                        train_actor = optimizer.apply_gradients(zip(grads_actor, self.actor_network.trainable_variables))
                        train_critic = optimizer.apply_gradients(zip(grads_critic, self.critic_network.trainable_variables))

                        sess.run(tf.global_variables_initializer())
                        sess.run([train_actor, train_critic])

                    self.actor_network.set_weights(global_actor_network.get_weights())
                    self.critic_network.set_weights(global_critic_network.get_weights())

            if self.worker_id == 0:
                print(f"Episode: {global_episode}, Total Reward: {episode_reward}")

# 创建 A3C 优化器
global_actor_network.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
global_critic_network.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# 创建 A3C 工作线程
workers = [Worker(worker_id=i, global_actor_network=global_actor_network, global_critic_network=global_critic_network) for i in range(num_workers)]

# 启动工作线程
for worker in workers:
    worker.start()

# 等待工作线程完成
for worker in workers:
    worker.join()
