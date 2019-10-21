import toml
import pprint

def parse_config(file_path):
    f = open(file_path, "r")
    toml_string = f.read()
    config_dict = toml.loads(toml_string)
    f.close()
    pprint.pprint(config_dict)
    return config_dict
