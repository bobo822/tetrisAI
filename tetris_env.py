import numpy as np
from tcp_client import TetrisTCPClient
import cv2

class TetrisEnvironment:
    """
    Tetris環境包裝器，提供標準的強化學習接口
    """
    def __init__(self, host='127.0.0.1', port=10612):
        self.client = TetrisTCPClient(host, port)
        self.action_space = 5  # 0: 左移, 1: 右移, 2: 逆時針旋轉, 3: 順時針旋轉, 4: 下落
        self.observation_space = 24  # 狀態特徵維度

        # 遊戲狀態
        self.current_state = None
        self.previous_lines = 0
        self.previous_height = 0
        self.previous_holes = 0
        self.step_count = 0

    def connect(self):
        """連接到遊戲伺服器"""
        return self.client.connect()

    def disconnect(self):
        """斷開連接"""
        self.client.disconnect()

    def reset(self):
        """重置環境"""
        if not self.client.connected:
            self.connect()

        # 開始新遊戲
        self.client.start_game()
        response = self.client.receive_response()

        if response is None:
            return np.zeros(self.observation_space)

        self.previous_lines = response['removed_lines']
        self.previous_height = response['height']
        self.previous_holes = response['holes']
        self.step_count = 0

        state = self._extract_features(response)
        self.current_state = state
        return state

    def step(self, action):
        """執行動作"""
        # 執行動作
        if action == 0:
            self.client.move(-1)
        elif action == 1:
            self.client.move(1)
        elif action == 2:
            self.client.rotate(0)
        elif action == 3:
            self.client.rotate(1)
        elif action == 4:
            self.client.drop()

        # 獲取響應
        response = self.client.receive_response()
        if response is None:
            return self.current_state, 0, True, {}

        # 提取狀態特徵
        next_state = self._extract_features(response)

        # 計算獎勵
        reward = self._calculate_reward(response, action)

        # 檢查遊戲是否結束
        done = response['is_over']

        # 更新狀態
        self.current_state = next_state
        self.previous_lines = response['removed_lines']
        self.previous_height = response['height']
        self.previous_holes = response['holes']
        self.step_count += 1

        info = {
            'removed_lines': response['removed_lines'],
            'height': response['height'],
            'holes': response['holes'],
            'step_count': self.step_count,
            'image': response['image']  # <--- 修正處：將遊戲畫面加入info字典
        }

        return next_state, reward, done, info

    def _extract_features(self, response):
        """從遊戲響應中提取特徵"""
        # 這裡需要根據實際的遊戲狀態數據來實現
        # 暫時使用示例特徵
        features = np.array([
            response['height'] / 22.0,  # 歸一化高度
            response['holes'] / 100.0,  # 歸一化洞數
            response['removed_lines'] / 1000.0,  # 歸一化消除行數
            self.step_count / 10000.0,  # 歸一化步數
        ])

        # 如果有圖像數據，可以進一步處理
        if response['image'] is not None:
            # 簡化的圖像處理
            img = cv2.resize(response['image'], (10, 20))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_features = img_gray.flatten() / 255.0
            # 只取前20個像素特徵
            features = np.concatenate([features, img_features[:20]])
        else:
            # 填充零特徵
            features = np.concatenate([features, np.zeros(20)])

        return features

    def _calculate_reward(self, response, action):
        """計算獎勵函數"""
        reward = 0

        remove_lines = response['removed_lines']

        # 消除行數獎勵
        lines_cleared = response['removed_lines'] - self.previous_lines
        reward += lines_cleared * 1000

        # 高度懲罰
        height_change = response['height'] - self.previous_height
        reward -= height_change * 5

        # 洞數懲罰
        holes_change = response['holes'] - self.previous_holes
        reward -= holes_change * 10

        # 下落動作小獎勵
        if action == 4:
            reward += 5

        # 生存獎勵
        reward += 1


        return reward

    def render(self):
        """渲染遊戲狀態（可選）"""
        pass