import os
import json
import numpy as np
import time
import pandas as pd
import argparse
from tqdm import tqdm
import torch

from tetris_env import TetrisEnvironment
from ddqn_agent import DDQNAgent
from genetic_optimizer import GeneticOptimizer
from game_recorder import GameRecorder, CSVLogger
from utils import create_directory, load_config, save_config

def train_ddqn(config, agent=None, env=None):
    """訓練DDQN智能體"""
    # 加載配置
    ddqn_config = config['ddqn']
    training_config = config['training']
    server_config = config['server']

    # 創建環境
    if env is None:
        env = TetrisEnvironment(server_config['host'], server_config['port'])
        env.connect()

    # 創建智能體
    if agent is None:
        agent = DDQNAgent(env.observation_space, env.action_space, ddqn_config)

    # 創建日誌記錄器
    create_directory('data')
    logger = CSVLogger('data/training_log.csv')

    # 訓練循環
    total_episodes = training_config['total_episodes']
    max_steps = training_config['max_steps_per_episode']

    print(f"開始訓練DDQN智能體，共 {total_episodes} 個回合...")

    best_reward = -float('inf')
    best_lines = 0

    for episode in tqdm(range(total_episodes)):
        # 重置環境
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        # 回合循環
        while not done and step < max_steps:
            # 選擇動作
            action = agent.select_action(state)

            # 執行動作
            next_state, reward, done, info = env.step(action)

            # 存儲經驗
            agent.remember(state, action, reward, next_state, done)

            # 更新狀態
            state = next_state
            total_reward += reward
            step += 1

            # 訓練智能體
            if len(agent.memory) > ddqn_config['batch_size']:
                loss = agent.replay()

        # 記錄日誌
        log_data = {
            'episode': episode,
            'reward': total_reward,
            'steps': step,
            'removed_lines': info.get('removed_lines', 0),
            'epsilon': agent.epsilon
        }
        logger.log(log_data)

        # 輸出進度
        if (episode + 1) % training_config['log_frequency'] == 0:
            print(f"回合 {episode+1}/{total_episodes} - 獎勵: {total_reward:.2f}, 步數: {step}, 消除行數: {info.get('removed_lines', 0)}, Epsilon: {agent.epsilon:.4f}")

        # 保存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            best_lines = info.get('removed_lines', 0)
            create_directory('models')
            agent.save('models/best_model.pth')
            print(f"發現新的最佳模型 - 獎勵: {best_reward:.2f}, 消除行數: {best_lines}")

        # 定期保存檢查點
        if (episode + 1) % training_config['save_frequency'] == 0:
            agent.save(f'models/checkpoint_{episode+1}.pth')

    # 保存訓練日誌
    logger.save()

    print(f"DDQN訓練完成! 最佳獎勵: {best_reward:.2f}, 最佳消除行數: {best_lines}")
    return agent, best_reward, best_lines

def train_with_genetic_algorithm(config):
    """使用遺傳算法優化DDQN超參數"""
    # 加載配置
    ga_config = config['genetic_algorithm']
    server_config = config['server']

    # 創建環境
    env = TetrisEnvironment(server_config['host'], server_config['port'])
    env.connect()

    # 創建遺傳算法優化器
    optimizer = GeneticOptimizer(
        pop_size=ga_config['population_size'],
        mutation_rate=ga_config['mutation_rate'],
        crossover_rate=ga_config['crossover_rate']
    )

    # 生成初始種群
    population = optimizer.generate_initial_population()

    # 創建日誌記錄器
    create_directory('data')
    logger = CSVLogger('data/ga_optimization_log.csv')

    print(f"開始遺傳算法優化，共 {ga_config['generations']} 代...")

    best_individual = None
    best_fitness = -float('inf')

    # 進化循環
    for generation in range(ga_config['generations']):
        print(f"第 {generation+1}/{ga_config['generations']} 代進化開始")

        # 評估種群適應度
        fitness_scores = []
        for i, hyperparams in enumerate(population):
            print(f"評估個體 {i+1}/{len(population)}")

            # 創建並訓練智能體
            agent = DDQNAgent(env.observation_space, env.action_space, hyperparams)

            # 簡短訓練
            _, reward, lines = train_ddqn(
                config,
                agent=agent,
                env=env
            )

            # 計算適應度
            fitness = reward + lines * 1000
            fitness_scores.append(fitness)

            # 記錄日誌
            log_data = {
                'generation': generation,
                'individual': i,
                'fitness': fitness,
                'reward': reward,
                'lines': lines
            }
            log_data.update(hyperparams)
            logger.log(log_data)

            # 更新最佳個體
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = hyperparams.copy()

                # 保存最佳個體
                save_config({
                    'best_hyperparams': best_individual,
                    'fitness': best_fitness,
                    'generation': generation
                })
                print(f"發現新的最佳超參數設置 - 適應度: {best_fitness:.2f}")

        # 進化種群
        population = optimizer.evolve(fitness_scores)

        # 打印當前最佳
        best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[best_idx]
        print(f"第 {generation+1} 代完成 - 當前最佳適應度: {gen_best_fitness:.2f}, 總體最佳適應度: {best_fitness:.2f}")

    # 保存日誌
    logger.save()

    print(f"遺傳算法優化完成! 最佳適應度: {best_fitness:.2f}")
    return best_individual


def evaluate_agent(config, model_path):
    """評估智能體性能"""
    # 加載配置
    server_config = config['server']
    eval_config = config['evaluation']

    # 創建環境
    env = TetrisEnvironment(server_config['host'], server_config['port'])
    env.connect()

    # 創建並加載智能體
    agent = DDQNAgent(env.observation_space, env.action_space, config['ddqn'])
    agent.load(model_path)
    agent.epsilon = 0.01  # 使用較低的ε進行評估

    # 創建錄製器和日誌記錄器
    recorder = None
    if eval_config.get('record_video', False):
        create_directory('recordings')
        recorder = GameRecorder(output_dir='recordings')

    create_directory('data')
    logger = CSVLogger('data/evaluation_results.csv')

    print(f"開始評估智能體，共 {eval_config['test_episodes']} 個回合...")

    total_lines_removed = 0
    total_steps = 0
    best_lines = 0
    best_steps = 0

    for episode in range(eval_config['test_episodes']):
        # 重置環境
        state = env.reset()
        done = False
        step = 0

        # 如果需要錄製，重置錄製器
        if eval_config.get('record_video', False) and recorder:
            recorder.reset()

        # 回合循環
        while not done:
            # 選擇動作
            action = agent.select_action(state, training=False)

            # 執行動作
            next_state, reward, done, info = env.step(action)

            # 錄製幀
            if eval_config.get('record_video', False) and recorder and 'image' in info:
                recorder.add_frame(info['image'], info)

            # 更新狀態
            state = next_state
            step += 1

        # 保存GIF
        if eval_config.get('record_video', False) and recorder:
            recorder.save(f"episode_{episode + 1}.gif")

        # 記錄結果
        lines_removed = info.get('removed_lines', 0)
        total_lines_removed += lines_removed
        total_steps += step

        # 更新最佳記錄
        if lines_removed > best_lines or (lines_removed == best_lines and step > best_steps):
            best_lines = lines_removed
            best_steps = step

        # 記錄日誌
        log_data = {
            'episode': episode,
            'removed_lines': lines_removed,
            'steps': step
        }
        logger.log(log_data)

        print(f"回合 {episode + 1}/{eval_config['test_episodes']} - 消除行數: {lines_removed}, 步數: {step}")

    # 保存評估日誌
    logger.save()

    # 計算平均結果
    avg_lines = total_lines_removed / eval_config['test_episodes']
    avg_steps = total_steps / eval_config['test_episodes']

    # 創建最終結果CSV
    if eval_config['save_csv']:
        result_df = pd.DataFrame([
            {'id': 0, 'removed_lines': best_lines, 'played_steps': best_steps},
            {'id': 1, 'removed_lines': best_lines, 'played_steps': best_steps}
        ])
        result_df.to_csv('data/tetris_best_score.csv', index=False)
        print(f"已保存最佳分數到 data/tetris_best_score.csv")

    print(f"評估完成! 平均消除行數: {avg_lines:.2f}, 平均步數: {avg_steps:.2f}")
    print(f"最佳結果 - 消除行數: {best_lines}, 步數: {best_steps}")
    print(f"自定義評分: {best_lines + best_steps / 1000000:.6f}")

    return best_lines, best_steps

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='DDQN+GA Tetris AI')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ga', 'evaluate'],
                        help='執行模式: train (訓練DDQN), ga (遺傳算法優化), evaluate (評估)')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路徑')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='模型文件路徑 (僅評估模式)')
    args = parser.parse_args()

    # 加載配置
    config = load_config(args.config)
    if torch.cuda.is_available():
        print(1)
    else:
        print(0)

    if args.mode == 'train':
        # 訓練DDQN
        train_ddqn(config)
    elif args.mode == 'ga':
        # 遺傳算法優化
        best_hyperparams = train_with_genetic_algorithm(config)
        # 更新配置
        config['ddqn'].update(best_hyperparams)
        save_config(config, args.config)
        # 使用優化後的超參數訓練
        train_ddqn(config)
    elif args.mode == 'evaluate':
        # 評估智能體
        evaluate_agent(config, args.model)

if __name__ == '__main__':
    main()
