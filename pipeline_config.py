from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    out_dir: Path
    checkpoints_dir: Path
    eval_dir: Path

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / 'last.pt'


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    digits: int
    paths: PipelinePaths
    collect_count: int
    variants_per_label: int
    train_epochs: int
    crop_right: int = 0

    def default_raw_dir(self) -> Path:
        return self.paths.raw_dir

    def default_processed_dir(self) -> Path:
        return self.paths.processed_dir

    def default_resume_path(self) -> Path:
        return self.paths.last_checkpoint_path


def build_pipeline_config(
    name: str,
    digits: int,
    collect_count: int,
    variants_per_label: int,
    train_epochs: int,
    crop_right: int = 0,
) -> PipelineConfig:
    data_dir = REPO_ROOT / 'data' / name
    out_dir = REPO_ROOT / 'out' / name
    paths = PipelinePaths(
        root=REPO_ROOT,
        data_dir=data_dir,
        raw_dir=data_dir / 'raw',
        processed_dir=data_dir / 'processed',
        out_dir=out_dir,
        checkpoints_dir=out_dir / 'checkpoints',
        eval_dir=out_dir / 'eval',
    )
    return PipelineConfig(
        name=name,
        digits=digits,
        paths=paths,
        collect_count=collect_count,
        variants_per_label=variants_per_label,
        train_epochs=train_epochs,
        crop_right=crop_right,
    )


PIPELINES: dict[str, PipelineConfig] = {
    'ccxp': build_pipeline_config(
        name='ccxp',
        digits=6,
        collect_count=50,
        variants_per_label=50,
        train_epochs=30,
        crop_right=13,
    ),
    'oauth': build_pipeline_config(
        name='oauth',
        digits=4,
        collect_count=1000,
        variants_per_label=10,
        train_epochs=10,
    ),
}
