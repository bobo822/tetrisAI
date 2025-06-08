import numpy as np
import random

class GeneticOptimizer:
    """
    遺傳算法超參數優化器，用於進化DDQN的超參數
    """
    def __init__(self, pop_size=10, mutation_rate=0.1, crossover_rate=0.8):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0

    def generate_initial_population(self):
        """生成初始超參數種群"""
        self.population = []
        for _ in range(self.pop_size):
            hyperparams = {
                'learning_rate': np.random.uniform(1e-5, 1e-3),
                'gamma': np.random.uniform(0.9, 0.999),
                'epsilon': np.random.uniform(0.1, 1.0),
                'epsilon_decay': np.random.uniform(0.995, 0.9999),
                'batch_size': np.random.choice([32, 64, 128, 256]),
                'update_frequency': np.random.choice([100, 200, 500, 1000]),
                'hidden_units': np.random.choice([64, 128, 256, 512]),
                'replay_buffer_size': np.random.choice([10000, 50000, 100000])
            }
            self.population.append(hyperparams)
        return self.population

    def evaluate_fitness(self, agent, env, episodes=10):
        """評估個體適應度"""
        total_reward = 0
        total_lines_removed = 0

        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            if 'removed_lines' in info:
                total_lines_removed += info['removed_lines']

        # 適應度函數結合了獎勵和消除行數
        fitness = total_reward + total_lines_removed * 1000
        return fitness / episodes

    def select_parents(self, fitness_scores):
        """選擇父母進行交叉"""
        # 確保適應度分數為正值
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            fitness_scores = [f - min_fitness + 1 for f in fitness_scores]

        fitness_sum = sum(fitness_scores)
        if fitness_sum == 0:
            selection_probs = [1/len(fitness_scores)] * len(fitness_scores)
        else:
            selection_probs = [f/fitness_sum for f in fitness_scores]

        parents_idx = np.random.choice(
            len(self.population), 
            size=2, 
            replace=False, 
            p=selection_probs
        )
        return parents_idx

    def crossover(self, parent1, parent2):
        """交叉操作產生後代"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy()

        child = {}
        for key in parent1.keys():
            if np.random.rand() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate(self, individual):
        """變異操作"""
        mutated = individual.copy()
        for key in mutated.keys():
            if np.random.rand() < self.mutation_rate:
                if key == 'learning_rate':
                    mutated[key] = np.random.uniform(1e-5, 1e-3)
                elif key == 'gamma':
                    mutated[key] = np.random.uniform(0.9, 0.999)
                elif key == 'epsilon':
                    mutated[key] = np.random.uniform(0.1, 1.0)
                elif key == 'epsilon_decay':
                    mutated[key] = np.random.uniform(0.995, 0.9999)
                elif key == 'batch_size':
                    mutated[key] = np.random.choice([32, 64, 128, 256])
                elif key == 'update_frequency':
                    mutated[key] = np.random.choice([100, 200, 500, 1000])
                elif key == 'hidden_units':
                    mutated[key] = np.random.choice([64, 128, 256, 512])
                elif key == 'replay_buffer_size':
                    mutated[key] = np.random.choice([10000, 50000, 100000])
        return mutated

    def evolve(self, fitness_scores):
        """進化過程"""
        new_population = []

        # 精英策略：保留最佳個體
        elite_count = max(1, self.pop_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # 生成新一代
        while len(new_population) < self.pop_size:
            parent_idx = self.select_parents(fitness_scores)
            parent1, parent2 = self.population[parent_idx[0]], self.population[parent_idx[1]]

            # 交叉
            child = self.crossover(parent1, parent2)

            # 變異
            child = self.mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation += 1
        return self.population

    def get_best_individual(self, fitness_scores):
        """獲取當前最佳個體"""
        best_idx = np.argmax(fitness_scores)
        return self.population[best_idx], fitness_scores[best_idx]
