from copy import deepcopy
from pathlib import Path
from typing import Any

import wandb
import yaml

from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import Modules
from neural_pfaffian.vmc import VMCState


class LoggerAbc:
    def _log_data(self, data: dict[str, Any]): ...

    def log_data(self, data: dict[str, Any], prefix: str | None = None):
        return self._log_data({f'{prefix}/{k}': v for k, v in data.items()})

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]: ...

    def config(self, config: dict[str, Any]): ...

    def checkpoint(self, state: VMCState, systems: Systems): ...

    def load_checkpoint(
        self, state: VMCState, systems: Systems
    ) -> tuple[VMCState, Systems]: ...

    def has_checkpoint(self) -> bool:
        return False


class WandbLogger(LoggerAbc):
    def __init__(self, **kwargs):
        self.run = wandb.init(**kwargs, resume='allow')

    def _log_data(self, data: dict[str, Any]):
        wandb.log(data)

    def update_config(self, config: dict[str, Any]):
        config = deepcopy(config)
        config['logging']['wandb'] = config['logging'].get('wandb', {}) | dict(
            id=self.run.id,
            project=self.run.project,
            entity=self.run.entity,
        )
        return config

    def config(self, config: dict[str, Any]):
        if next(iter(config.keys())) not in self.run.config:
            self.run.config.update(config)

    def checkpoint(self, state: VMCState, systems: Systems):
        raise NotImplementedError

    def load_checkpoint(self, state: VMCState, systems: Systems):
        raise NotImplementedError


class LogFile:
    def __init__(self, path: Path | str, delimiter: str = ','):
        self.path = Path(path)
        if self.path.exists():
            headers = self.path.open('r').readline().strip().split(',')
            if len(headers) == 0 or headers == ['']:
                headers = None
        else:
            headers = None
        self.headers = headers
        self._logfile = open(self.path, 'a')

    def write(self, data: dict[str, Any]):
        if self.headers is None:
            self.headers = list(data.keys())
            self._logfile.write(','.join(self.headers) + '\n')
        self._logfile.write(','.join(str(data.get(h, '')) for h in self.headers) + '\n')
        self._logfile.flush()


class FileLogger(LoggerAbc):
    def __init__(self, directory: Path | str, delimiter: str = ','):
        assert directory is not None
        self.directory = Path(directory).resolve().absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.delimiter = delimiter
        self._log_files: dict[str, LogFile] = {}

    @property
    def config_path(self):
        return self.directory / 'config.yaml'

    @property
    def state_path(self):
        return self.directory / 'state.msgpack'

    @property
    def systems_path(self):
        return self.directory / 'systems.msgpack'

    def logfile_path(self, prefix: str):
        return self.directory / f'{prefix}_log.csv'

    def logfile(self, prefix: str):
        return self._log_files.setdefault(
            prefix, LogFile(self.logfile_path(prefix), self.delimiter)
        )

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        config = deepcopy(config)
        config['logging']['file'] = config['logging'].get('file', {}) | dict(
            directory=str(self.directory)
        )
        return config

    def config(self, config: dict[str, Any]):
        self.config_path.write_text(yaml.dump(config))

    def log_data(self, data: dict[str, Any], prefix: str | None = None):
        if prefix is None:
            prefix = 'main'
        self.logfile(prefix).write(data)

    def checkpoint(self, state: VMCState, systems: Systems):
        state.to_file(self.state_path)
        systems.to_file(self.systems_path)

    def load_checkpoint(self, state: VMCState, systems: Systems):
        return state.from_file(self.state_path), systems.from_file(self.systems_path)

    def has_checkpoint(self) -> bool:
        return self.state_path.exists() and self.systems_path.exists()


class Logger:
    def __init__(self, logging_config):
        self.loggers = LOGGERS.try_init_many(logging_config)

    def log(self, data: dict[str, Any], prefix: str | None = None):
        for logger in self.loggers:
            logger.log_data(data, prefix)

    def config(self, config: dict[str, Any]):
        config = {k.replace('_config', ''): v for k, v in config.items()}
        for logger in self.loggers:
            config = logger.update_config(config)
        for logger in self.loggers:
            logger.config(config)

    def checkpoint(self, state: VMCState, systems: Systems):
        for logger in self.loggers:
            try:
                logger.checkpoint(state, systems)
            except NotImplementedError:
                pass

    def load_checkpoint(self, state: VMCState, systems: Systems):
        for logger in self.loggers:
            if logger.has_checkpoint():
                return logger.load_checkpoint(state, systems)
        return state, systems

    def has_checkpoint(self) -> bool:
        return any(logger.has_checkpoint() for logger in self.loggers)


LOGGERS = Modules[LoggerAbc](
    {cls.__name__.lower().replace('logger', ''): cls for cls in [WandbLogger, FileLogger]}
)
