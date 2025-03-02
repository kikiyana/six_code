import numpy as np
import tensorflow as tf
import gym

# 定义深度确定性策略梯度（DDPG）网络
class DDPGNetwork(tf.keras.Model):
    def __init__(self, name):
        super(DDPGNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name=name + '_dense1')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', name=name + '_dense2')
        self.output_layer = tf.keras.layers.Dense(1, activation='tanh', name=name + '_output')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# 定义DDPG智能体
class DDPGAgent:
    def __init__(self, name, state_dim, action_dim):
        self.actor = DDPGNetwork(name + '_actor')
        self.critic = DDPGNetwork(name + '_critic')
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_action(self, state):
        return self.actor(state)

    def train(self, states, actions, rewards, next_states, gamma=0.99):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            target_actions = self.actor(next_states)
            q_values_next = self.critic(tf.concat([next_states, target_actions], axis=-1))
            q_values = self.critic(tf.concat([states, actions], axis=-1))
            target_q_values = rewards + gamma * q_values_next
            actor_loss = -tf.reduce_mean(q_values)
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

# 创建两个DDPG智能体
agent1 = DDPGAgent('agent1', state_dim=3, action_dim=1)
agent2 = DDPGAgent('agent2', state_dim=3, action_dim=1)

# 创建环境
env = gym.make('Pendulum-v0')

# 训练MARL智能体
episodes = 5000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action1 = agent1.get_action(state[np.newaxis, :])
        action2 = agent2.get_action(state[np.newaxis, :])
        actions = np.concatenate([action1, action2], axis=-1)
        next_state, reward, done, _ = env.step(actions[0])
        agent1.train(state, actions, reward, next_state)
        agent2.train(state, actions, reward, next_state)
        total_reward += reward
        state = next_state

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# 关闭环境
env.close()
