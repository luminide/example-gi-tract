import os
import yaml

class Config():
    def __init__(self, init=None):
        if init is None:
            init = Config.get_defaults()
        object.__setattr__(self, "_params", dict())
        self.update(init)

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, val):
        self._params[key] = val

    def __getattr__(self, key):
        return self._params[key]

    def __str__(self):
        return yaml.dump(self._params)

    def as_dict(self):
        return self._params

    def update(self, init):
        for key in init:
            self[key] = init[key]

    @staticmethod
    def get_defaults():
        conf_file = '../code/config.yaml'
        assert os.path.exists(conf_file)
        with open(conf_file) as file:
            return yaml.safe_load(file)
