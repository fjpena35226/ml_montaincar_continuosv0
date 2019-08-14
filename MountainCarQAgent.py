import gym
import numpy as np
import math
from tqdm import trange
from bayes_opt import BayesianOptimization
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class MountainCarQAgent:
    def __init__(self, action_bucket = 0.2, num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        self.action_bucket = action_bucket
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('MountainCarContinuous-v0')

        # [position, velocity]
        self.upper_bounds = self.env.high_state
        self.lower_bounds = self.env.low_state

        num_states = (self.env.observation_space.high - self.env.observation_space.low) * np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        #self.actions = np.arange(self.env.min_action,self.env.max_action, action_bucket)
        can_actions = (self.env.max_action - self.env.min_action) / action_bucket
        self.actions = np.linspace(self.env.min_action,self.env.max_action, np.round(can_actions, 0).astype(int))

        #self.Q_table = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], len(self.actions)))
        self.Q_table = np.zeros((num_states[0], num_states[1], len(self.actions)))

    def discretize_state(self, obs):
        state2_adj = (obs - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state2_adj, 0).astype(int)

    def discretize_action(self, acc):
        idx = (np.abs(self.actions - acc)).argmin()
        return [self.actions[idx]]

    def index_action(self, acc):
        return (np.abs(self.actions - acc)).argmin()

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.choose_best_action(state)

    def choose_best_action(self, state):
        indx = np.argmax(self.Q_table[state[0], state[1]])
        acc = self.actions[indx]
        if acc < 0:
            return [np.random.uniform(self.env.min_action,acc)] #si la mejor accion es negativa hay que tomar impulso
        else:
            return [np.random.uniform(acc, self.env.max_action)] # si la mejor accion es positiva hay que acelerar

    def update_q(self, state, action, reward, new_state):
        indx_action = self.index_action(action)

        actual_reward = self.Q_table[state[0], state[1], indx_action]

        self.Q_table[state[0], state[1], indx_action] += \
            self.learning_rate * (
                reward + self.discount * np.max(self.Q_table[new_state[0], new_state[1]]) - self.Q_table[state[0], state[1], indx_action])

        updated_reward =  self.Q_table[state[0], state[1], indx_action]

        if updated_reward > actual_reward:
            self.actions[indx_action] = action[0]

        return self.Q_table[state[0], state[1], indx_action]

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        steps = []
        for e in trange(self.num_episodes):
            # track the total time steps in this episode
            time = 0

            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                #if e >= (self.num_episodes - 3):
                #    self.env.render()
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                time += self.update_q(current_state, action, reward, new_state)
                current_state = new_state

            steps.append(time)
            #print("episode " + format(e) + " epsilon: " + format(self.epsilon))

        #self.savePerformance(steps)

    def savePerformance(self, steps):
        steps = np.add.accumulate(steps)
        plt.plot(steps, np.arange(1, len(steps) + 1))
        plt.xlabel('Reward accumulated')
        plt.ylabel('Episodes')

        plt.savefig('figure_QAgCartPolePerformance.png')
        plt.close()

    def run(self, render):
        done = False
        current_state = self.discretize_state(self.env.reset())
        states = list()
        while not done:
            if render:
                self.env.render()
            action = self.choose_best_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            new_state = self.discretize_state(obs)
            #print(format(current_state) + " -> " + format(action) + " -> " + format(new_state) + " (w=" +  format(reward) + ")")
            current_state = new_state
            states.append(obs)
        return states

def solve_mountaincar_agent(action_bucket=0.01, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25, render = bool(0), train=1, its = 50):
    c = 0  # cantidad de veces que resuelve el problema
    total_res = train * its;
    for t in range(0, train):
        agent = MountainCarQAgent(action_bucket, 1000, min_lr, min_epsilon, discount, decay)
        agent.train()
        #print(agent.actions)
        for i in range(0,total_res):
            states = agent.run(render)
            #print(states.pop()[0])
            if states.pop()[0] >= 0.45: #si pudo pasar/llegar 0.45 resolvio el problema
               c = c + 1
    return c/total_res# taza de resolucion

def optimization():
    hypScope = {
        'action_bucket': (0.1, 0.6)
    }

    bo = BayesianOptimization(solve_mountaincar_agent, hypScope)

    bo.maximize()

    print(bo.max)

if __name__ == "__main__":

   #optimization()

   taza_resolucion = solve_mountaincar_agent(0.5, 0.01, 0.1,  1.00, 29, bool(1),1,5)
   print(taza_resolucion)
