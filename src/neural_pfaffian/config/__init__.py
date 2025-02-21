import sys
import types
from pathlib import Path

config_root = Path(__file__).parent

# Imports all *.yaml files in the config directory as pseudo-modules
# Forces SEML to upload them to the database
if __package__ is not None:
    for config_file in Path.glob(config_root, '**/*.yaml'):
        config_module = types.ModuleType(config_file.stem)
        config_module.__file__ = str(config_file)
        sys.modules[__package__ + '.' + config_file.stem] = config_module

DEFAULT_CONFIG = str(Path(__file__).parent / 'default.yaml')
