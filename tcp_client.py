import socket
import numpy as np
import cv2
import struct

class TetrisTCPClient:
    """
    Tetris TCP客戶端，用於與Java伺服器通信
    """
    def __init__(self, host='127.0.0.1', port=10612):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """連接到Tetris伺服器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"已連接到伺服器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"連接失敗: {e}")
            return False

    def disconnect(self):
        """斷開連接"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("已斷開連接")

    def send_command(self, command):
        """發送命令到伺服器"""
        if not self.connected:
            return False
        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.socket.sendall(command_bytes)
            return True
        except Exception as e:
            print(f"發送命令失敗: {e}")
            return False

    def receive_response(self):
        """接收伺服器響應"""
        if not self.connected:
            return None

        try:
            # 讀取遊戲狀態
            is_over = self.socket.recv(1)[0] == 1
            removed_lines = struct.unpack('>I', self.socket.recv(4))[0]
            height = struct.unpack('>I', self.socket.recv(4))[0]
            holes = struct.unpack('>I', self.socket.recv(4))[0]
            img_size = struct.unpack('>I', self.socket.recv(4))[0]

            # 讀取PNG圖像
            img_data = b''
            while len(img_data) < img_size:
                chunk = self.socket.recv(img_size - len(img_data))
                if not chunk:
                    break
                img_data += chunk

            # 解碼圖像
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return {
                'is_over': is_over,
                'removed_lines': removed_lines,
                'height': height,
                'holes': holes,
                'image': image
            }
        except Exception as e:
            print(f"接收響應失敗: {e}")
            return None

    def start_game(self):
        """開始遊戲"""
        return self.send_command("start")

    def move(self, direction):
        """移動方塊 (-1: 左, 1: 右)"""
        return self.send_command(f"move {direction}")

    def rotate(self, direction):
        """旋轉方塊 (0: 逆時針, 1: 順時針)"""
        return self.send_command(f"rotate {direction}")

    def drop(self):
        """下落方塊"""
        return self.send_command("drop")
