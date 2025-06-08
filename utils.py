import os
import json
import numpy as np


def create_directory(directory):
    """創建目錄"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"創建目錄: {directory}")

def load_config(filepath):
    """加載配置文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"已加載配置: {filepath}")
        return config
    except Exception as e:
        print(f"加載配置失敗: {e}")
        return {}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_config(config, filename = 'models/best_hyperparams.json'):
    import json
    import os
    os.makedirs('models', exist_ok=True)  # 確保目錄存在
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)


def format_time(seconds):
    """格式化時間"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
