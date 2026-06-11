import atexit
import contextlib
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import yaml

import wandb
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import Modules
from neural_pfaffian.vmc import VMCState


class LoggerAbc:
    def __init__(self, run_name: str, **_): ...

    def _log_data(self, data: dict[str, Any]): ...

    def log_data(self, data: dict[str, Any], prefix: str | None = None):
        if prefix is None:
            return self._log_data(data)
        return self._log_data({f'{prefix}/{k}': v for k, v in data.items()})

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]: ...

    def config(self, config: dict[str, Any]): ...

    def checkpoint(self, state: VMCState, systems: Systems): ...

    def load_checkpoint(
        self,
        state: VMCState,
        systems: Systems,
    ) -> tuple[VMCState, Systems]: ...

    def has_checkpoint(self) -> bool:
        return False


class WandbLogger(LoggerAbc):
    def __init__(self, run_name: str, **kwargs):
        config: dict[str, Any] = {'name': run_name} | kwargs
        self.run = wandb.init(**config, resume='allow')

    def _log_data(self, data: dict[str, Any]):
        wandb.log(data)

    def update_config(self, config: dict[str, Any]):
        config = deepcopy(config)
        config['logging']['wandb'] = config['logging'].get('wandb', {}) | {
            'id': self.run.id,
            'project': self.run.project,
            'entity': self.run.entity,
            'name': self.run.name,
        }
        return config

    def config(self, config: dict[str, Any]):
        if next(iter(config.keys())) not in self.run.config:
            self.run.config.update(config)

    def checkpoint(self, state: VMCState, systems: Systems):
        raise NotImplementedError

    def load_checkpoint(self, state: VMCState, systems: Systems):
        raise NotImplementedError


class CsvLogFile:
    def __init__(self, path: Path | str, delimiter: str = ','):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            headers = self.path.open('r').readline().strip().split(',')
            if len(headers) == 0 or headers == ['']:
                headers = None
        else:
            headers = None
        self.headers = headers
        self.delimiter = delimiter
        self._logfile = open(self.path, 'a')  # noqa: SIM115

        atexit.register(self.close)

    def write(self, data: dict[str, Any]):
        if self.headers is None:
            self.headers = list(data.keys())
            self._logfile.write(self.delimiter.join(self.headers) + '\n')
        self._logfile.write(
            self.delimiter.join(str(data.get(h, '')) for h in self.headers) + '\n',
        )
        self._logfile.flush()

    def close(self):
        if not self._logfile.closed:
            self._logfile.close()

    def __del__(self):
        if not self._logfile.closed:
            self._logfile.close()


class FileLogger(LoggerAbc):
    def __init__(
        self,
        run_name: str,
        base_dir: Path | str = '.',
        directory: Path | str | None = None,
        delimiter: str = ',',
    ):
        if directory is None:
            directory = Path(base_dir) / str(
                run_name
                + datetime.now().strftime(
                    '_%Y-%m-%d_%H-%M-%S',
                ),
            )

        self.directory = Path(directory).resolve().absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.delimiter = delimiter
        self._csv_log_files: dict[str, CsvLogFile] = {}

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

    def csv_logfile(self, prefix: str):
        if prefix not in self._csv_log_files:
            self._csv_log_files[prefix] = CsvLogFile(
                self.logfile_path(prefix),
                self.delimiter,
            )
        return self._csv_log_files[prefix]

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        config = deepcopy(config)
        config['logging']['file'] = config['logging'].get('file', {}) | {
            'directory': str(self.directory),
        }
        return config

    def config(self, config: dict[str, Any]):
        self.config_path.write_text(yaml.dump(config))

    def log_data(self, data: dict[str, Any], prefix: str | None = None):
        if prefix is None:
            prefix = 'main'
        self.csv_logfile(prefix).write(data)

    def checkpoint(self, state: VMCState, systems: Systems):
        state.to_file(self.state_path)
        systems.to_file(self.systems_path)

    def load_checkpoint(self, state: VMCState, systems: Systems):
        return state.from_file(self.state_path), systems.from_file(self.systems_path)

    def has_checkpoint(self) -> bool:
        return self.state_path.exists() and self.systems_path.exists()


class Logger:
    def __init__(
        self,
        system_name: str,
        logging_config,
    ):
        config = deepcopy(logging_config)

        # Generate a unique, timestamped name for the run
        run_name = f'{system_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        # Update the logging config with the run name
        if isinstance(logging_config, dict):
            config = {k: v | {'run_name': run_name} for k, v in config.items()}
        elif isinstance(logging_config, Sequence):
            config = [
                (module[0], module[1] | {'run_name': run_name}) for module in config
            ]

        self.loggers = LOGGERS.try_init_many(config)

    def log(self, data: dict[str, Any], prefix: str | None = None):
        data = jax.device_get(data)
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
            with contextlib.suppress(NotImplementedError):
                logger.checkpoint(state, systems)

    def load_checkpoint(self, state: VMCState, systems: Systems):
        for logger in self.loggers:
            if logger.has_checkpoint():
                return logger.load_checkpoint(state, systems)
        return state, systems

    def has_checkpoint(self) -> bool:
        return any(logger.has_checkpoint() for logger in self.loggers)


LOGGERS = Modules[LoggerAbc](
    {
        cls.__name__.lower().replace('logger', ''): cls
        for cls in [WandbLogger, FileLogger]
    },
)
