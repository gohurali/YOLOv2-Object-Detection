from ruamel.yaml import YAML
def open_config():
    yaml = YAML()
    config = yaml.load(open('config.yaml'))
    return config