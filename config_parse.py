import toml

def parse_config(file_path):
    f = open(file_path, "r")
    toml_string = f.read()
    config_dict = toml.loads(toml_string)
    f.close()
    return config_dict
