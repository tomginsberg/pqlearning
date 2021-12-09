from typing import Dict, Any


def update_functional(args: Dict[str, Any]):
    if args is None:
        return lambda x, y: x

    def f(default: Any, name: str):
        if name in args:
            return args[name]
        return default

    return f


class no_train(object):
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.module.train(False)

    def __exit__(self):
        self.module.train(True)
