import csv
import sys
import logging
import datetime
from pathlib import Path
from typing import Dict, Any


class Logger:
    _COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "grey": "\033[90m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
    }
    _instance = None
    

    def __init__(
        self,
        log_dir: str = "experiments",
        experiment_name: str = "xgboost",
        use_tensorboard: bool = False,
        csv_filename: str = "metrics.csv",
    ):
        if hasattr(self, "_initialized"):
            return
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        log_file = self.log_dir / f"run_{timestamp}.log"
        self._file_logger = logging.getLogger(f"{experiment_name}_{timestamp}")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.propagate = False

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._file_logger.addHandler(file_handler)

        self._csv_path = self.log_dir / csv_filename
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = None
        self._csv_headers_written = False

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
                self._console("info", f"TensorBoard → {tb_dir}")
            except ImportError:
                self._console(
                    "warning",
                    "TensorBoard not installed — skipping. "
                    "Run: pip install tensorboard",
                )

        self.info(f"Logger initialized — {self.log_dir}")
        self.info(f"Log file - {log_file}")
        self.info(f"CSV metrics - {self._csv_path}")
        
        self._initialized = True

    def _color(self, text: str, *color_names: str) -> str:
        """Wrap text in ANSI codes. Skipped if not a TTY (e.g. redirected to file)."""
        if not sys.stdout.isatty():
            return text
        prefix = "".join(self._COLORS.get(c, "") for c in color_names)
        return f"{prefix}{text}{self._COLORS['reset']}"

    def _console(self, level: str, message: str):
        """Print a formatted line to console."""
        now = datetime.datetime.now().strftime("%H:%M:%S")

        level_formats = {
            "info": (self._color("INFO   ", "green", "bold"), ""),
            "warning": (self._color("WARNING", "yellow", "bold"), ""),
            "error": (self._color("ERROR  ", "red", "bold"), ""),
            "metric": (self._color("METRIC ", "cyan", "bold"), ""),
        }
        level_str, _ = level_formats.get(level, ("LOG    ", ""))
        time_str = self._color(now, "grey")
        print(f"{time_str} | {level_str} | {message}")

    def info(self, message: str):
        self._console("info", message)
        self._file_logger.info(message)

    def warning(self, message: str):
        self._console("warning", message)
        self._file_logger.warning(message)

    def error(self, message: str):
        self._console("error", message)
        self._file_logger.error(message)

    def log_metrics(
        self,
        epoch: int,
        split: str,
        metrics: Dict[str, Any],
        verbose: bool = True,
    ):
        """
        Log a dict of metrics for one epoch/split.

        Args:
            epoch:   current epoch number
            split:   "train" | "val" | "test"
            metrics: dict of {metric_name: value}, e.g. {"loss": 0.42, "mae": 0.31}
            verbose: if True, print to console
        """
        if verbose:
            parts = "  ".join(
                (
                    f"{k}={self._color(f'{v:.4f}', 'bold')}"
                    if isinstance(v, float)
                    else f"{k}={v}"
                )
                for k, v in metrics.items()
            )
            self._console("metric", f"[epoch {epoch:03d}] [{split}]  {parts}")

        parts_plain = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self._file_logger.info(f"[epoch {epoch:03d}] [{split}]  {parts_plain}")

        row = {"epoch": epoch, "split": split, **metrics}
        if not self._csv_headers_written:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
            self._csv_headers_written = True
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(f"{split}/{k}", v, epoch)

    def log_config(self, cfg):
        """Dump the full config object to the log file at run start."""
        self.info("=" * 60)
        self.info("Config:")
        for section_name in vars(cfg):
            section = getattr(cfg, section_name)
            self.info(f"  [{section_name}]")
            for k, v in vars(section).items():
                self.info(f"    {k}: {v}")
        self.info("=" * 60)

    def close(self):
        """Call at the end of training to flush and close all writers."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
        self.info("Logger closed.")


logger = Logger()