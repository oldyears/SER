import argparse
import yaml

class Config:
    """dict -> Class"""
    def __init__(self, entries: dict={}):
        for k, v in entries.items():
            if k != 'params' and isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v


def load_config(file_path: str) -> dict:
    """
    从 YAML 文件中加载配置

    Args:
        file_path (str): 配置文件路径

    Returns:
        config (dict): 配置项
    """
    f = open(file_path, 'r', encoding = 'utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def parse_opt():
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        '--config',
        type = str,
        default = 'configs/lstm.yaml',
        help = 'path to the configuration file (yaml)'
    )
    parser.add_argument(
        '--file_path',
        type = str,
    )

    args = parser.parse_args()
    file_path = args.file_path
    config_dict = load_config(args.config)
    config = Config(config_dict)

    return config,file_path
