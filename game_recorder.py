import cv2
import numpy as np
import os
import io
import imageio
from PIL import Image


class GameRecorder:
    """
    遊戲錄製器，用於記錄AI遊戲過程為GIF
    """

    def __init__(self, output_dir='recordings'):
        self.output_dir = output_dir
        self.frames = []
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def reset(self):
        """重置錄製器，清空所有幀"""
        self.frames = []

    def add_frame(self, frame_data, game_info=None):
        """
        添加一幀到錄製中，並可選擇性疊加遊戲信息
        """
        # 如果frame_data是PNG字節數據
        if isinstance(frame_data, bytes):
            image = Image.open(io.BytesIO(frame_data))
            frame = np.array(image)
        else:
            frame = frame_data.copy()  # 使用副本以避免修改原始數據

        # 疊加遊戲信息
        if game_info:
            self._add_game_info_overlay(frame, game_info)

        self.frames.append(frame)

    def save(self, filename, fps=10):
        """將所有幀保存為GIF文件"""
        if self.frames:
            filepath = os.path.join(self.output_dir, filename)

            # imageio預期使用RGB格式，但cv2提供BGR，因此需進行顏色轉換
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in self.frames]

            # 使用imageio保存為gif
            imageio.mimsave(filepath, rgb_frames, fps=fps)
            print(f"GIF錄製完成: {filepath}")
            self.reset()  # 保存後清空
            return filepath
        return None

    def _add_game_info_overlay(self, image, game_info):
        """在圖像上添加遊戲信息覆蓋"""
        # --- 修改開始 ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # 縮小字體
        color = (255, 255, 255)  # 白色
        thickness = 1  # 變細字體

        x_pos = 5
        y_offset = 15  # 從更靠近頂部的位置開始
        y_increment = 15  # 縮小行距
        # --- 修改結束 ---

        # 添加文字信息
        if 'removed_lines' in game_info:
            text = f"Lines: {game_info['removed_lines']}"
            cv2.putText(image, text, (x_pos, y_offset), font, font_scale, color, thickness)
            y_offset += y_increment

        if 'height' in game_info:
            text = f"Height: {game_info['height']}"
            cv2.putText(image, text, (x_pos, y_offset), font, font_scale, color, thickness)
            y_offset += y_increment

        if 'holes' in game_info:
            text = f"Holes: {game_info['holes']}"
            cv2.putText(image, text, (x_pos, y_offset), font, font_scale, color, thickness)
            y_offset += y_increment

        if 'step_count' in game_info:
            text = f"Steps: {game_info['step_count']}"
            cv2.putText(image, text, (x_pos, y_offset), font, font_scale, color, thickness)


class CSVLogger:
    """
    CSV日誌記錄器，用於記錄訓練和評估數據
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.logs = []

    def log(self, data):
        """記錄數據"""
        self.logs.append(data)

    def save(self):
        """保存到CSV文件"""
        if not self.logs:
            return

        import pandas as pd
        df = pd.DataFrame(self.logs)
        df.to_csv(self.filepath, index=False)
        print(f"日誌保存到: {self.filepath}")

    def clear(self):
        """清空日誌"""
        self.logs = []