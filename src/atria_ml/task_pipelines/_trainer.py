from collections.abc import Callable
from functools import partial
from typing import ClassVar

import hydra_zen
from atria_core.logger import get_logger
from atria_core.types import TaskType
from atria_datasets.pipelines.atria_data_pipeline import AtriaDataPipeline
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from omegaconf import OmegaConf

from atria_ml.registry import TASK_PIPELINE
from atria_ml.task_pipelines._inferencer import Inferencer
from atria_ml.training.engines.evaluation import (
    InferenceEngine,
    TestEngine,
    ValidationEngine,
    VisualizationEngine,
)
from atria_ml.training.engines.training import TrainingEngine
from atria_ml.training.engines.utilities import InferenceConfig, RunConfig

logger = get_logger(__name__)

_DEFAULT_ALLOWED_KEYS = ["index", "sample_id", "page_id", "total_num_pages", "gt"]


@TASK_PIPELINE.register(
    "trainer",
    zen_meta={"n_devices": 1, "backend": "nccl", "experiment_name": hydra_zen.MISSING},
    zen_exclude=["hydra", "package", "version"],
    is_global_package=True,
)
class Trainer:
    _REGISTRY_CONFIGS: ClassVar[dict] = {
        "default": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                "_self_",
            ],
            "output_dir": "outputs/trainer/image_classification/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
        },
        "image_classification": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/model_pipeline@model_pipeline": "image_classification"},
                {"/metric_factory@metric_factory.accuracy": "accuracy"},
                {"/metric_factory@metric_factory.precision": "precision"},
                {"/metric_factory@metric_factory.recall": "recall"},
                {"/metric_factory@metric_factory.f1_score": "f1_score"},
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                {
                    "/data_transform@data_pipeline.runtime_transforms.train": "image_transform/default"
                },
                {
                    "/data_transform@data_pipeline.runtime_transforms.evaluation": "image_transform/default"
                },
                "_self_",
            ],
            "task_type": TaskType.image_classification,
            "data_pipeline": {"allowed_keys": _DEFAULT_ALLOWED_KEYS + ["image"]},
            "output_dir": "outputs/trainer/image_classification/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
        },
        "sequence_classification": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/model_pipeline@model_pipeline": "sequence_classification"},
                {"/metric_factory@metric_factory.accuracy": "accuracy"},
                {"/metric_factory@metric_factory.precision": "precision"},
                {"/metric_factory@metric_factory.recall": "recall"},
                {"/metric_factory@metric_factory.f1_score": "f1_score"},
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                {
                    "/data_transform@data_pipeline.runtime_transforms.train": "document_instance_tokenizer/sequence_classification"
                },
                {
                    "/data_transform@data_pipeline.runtime_transforms.evaluation": "document_instance_tokenizer/sequence_classification"
                },
                "_self_",
            ],
            "task_type": TaskType.sequence_classification,
            "data_pipeline": {"allowed_keys": _DEFAULT_ALLOWED_KEYS + ["image"]},
            "output_dir": "outputs/trainer/sequence_classification/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
        },
        "semantic_entity_recognition": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/model_pipeline@model_pipeline": "token_classification"},
                {"/metric_factory@metric_factory.accuracy": "seqeval_accuracy_score"},
                {"/metric_factory@metric_factory.precision": "seqeval_precision_score"},
                {"/metric_factory@metric_factory.recall": "seqeval_recall_score"},
                {"/metric_factory@metric_factory.f1_score": "seqeval_f1_score"},
                {
                    "/metric_factory@metric_factory.classification_report": "seqeval_classification_report"
                },
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                {
                    "/data_transform@data_pipeline.runtime_transforms.train": "document_instance_tokenizer/semantic_entity_recognition"
                },
                {
                    "/data_transform@data_pipeline.runtime_transforms.evaluation": "document_instance_tokenizer/semantic_entity_recognition"
                },
                "_self_",
            ],
            "task_type": TaskType.semantic_entity_recognition,
            "data_pipeline": {"allowed_keys": _DEFAULT_ALLOWED_KEYS + ["image"]},
            "output_dir": "outputs/trainer/semantic_entity_recognition/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
        },
        "layout_entity_recognition": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/model_pipeline@model_pipeline": "layout_token_classification"},
                {"/metric_factory@metric_factory.precision": "layout_precision"},
                {"/metric_factory@metric_factory.recall": "layout_recall"},
                {"/metric_factory@metric_factory.f1_score": "layout_f1"},
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                {
                    "/data_transform@data_pipeline.runtime_transforms.train": "document_instance_tokenizer/semantic_entity_recognition"
                },
                {
                    "/data_transform@data_pipeline.runtime_transforms.evaluation": "document_instance_tokenizer/semantic_entity_recognition"
                },
                "_self_",
            ],
            "task_type": TaskType.layout_entity_recognition,
            "data_pipeline": {"allowed_keys": _DEFAULT_ALLOWED_KEYS + ["image"]},
            "output_dir": "outputs/trainer/layout_entity_recognition/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
        },
        "visual_question_answering": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/model_pipeline@model_pipeline": "question_answering"},
                {"/metric_factory@metric_factory.sequence_anls": "sequence_anls"},
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                {
                    "/data_transform@data_pipeline.runtime_transforms.train": "document_instance_tokenizer/visual_question_answering"
                },
                {
                    "/data_transform@data_pipeline.runtime_transforms.evaluation": "document_instance_tokenizer/visual_question_answering"
                },
                "_self_",
            ],
            "task_type": TaskType.visual_question_answering,
            "data_pipeline": {"allowed_keys": _DEFAULT_ALLOWED_KEYS + ["image"]},
            "output_dir": "outputs/trainer/visual_question_answering/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
        },
        "layout_analysis": {
            "hydra_defaults": [
                {"/data_pipeline@data_pipeline": "default"},
                {"/model_pipeline@model_pipeline": "object_detection"},
                {"/metric_factory@metric_factory.cocoeval": "cocoeval"},
                {"/engine@training_engine": "default_training_engine"},
                {"/engine@validation_engine": "default_validation_engine"},
                {"/engine@test_engine": "default_test_engine"},
                {"/engine@visualization_engine": "default_visualization_engine"},
                {
                    "/data_transform@data_pipeline.runtime_transforms.train": "document_instance_mmdet_transform/train"
                },
                {
                    "/data_transform@data_pipeline.runtime_transforms.evaluation": "document_instance_mmdet_transform/evaluation"
                },
                {
                    "override /optimizer_factory@training_engine.optimizer_factory": "sgd"
                },
                {
                    "override /lr_scheduler_factory@training_engine.lr_scheduler_factory": "multi_step_lr"
                },
                "_self_",
            ],
            "task_type": TaskType.layout_analysis,
            "data_pipeline": {
                "allowed_keys": _DEFAULT_ALLOWED_KEYS + ["image"],
                "collate_fn": "mmdet_pseudo_collate",
            },
            "training_engine": {
                "model_checkpoint_config": {"monitored_metric": None},
                "optimizer_factory": {
                    "lr": 0.02,
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                },
                "lr_scheduler_factory": {"milestones": [16, 22], "gamma": 0.1},
                "warmup_config": {"warmup_ratio": 0.0, "warmup_steps": 500},
            },
            "output_dir": "outputs/trainer/layout_analysis/${data_pipeline.dataset.dataset_name}/${model_pipeline.model.name}/${experiment_name}",
            "do_visualization": True,
            "visualization_engine": {
                "visualize_every_n_epochs": 10,
                "visualize_on_start": False,
            },
            "validation_engine": {
                "validate_every_n_epochs": 10,
                "validate_on_start": False,
            },
        },
        # generative_modeling=dict(
        #     hydra_defaults=[
        #         "_self_",
        #         {"/metric_factory@metric_factory.fid_score": "fid_score"},
        #     ],
        # ),
    }

    def __init__(
        self,
        task_type: TaskType,
        data_pipeline: AtriaDataPipeline,
        model_pipeline: AtriaModelPipeline,
        training_engine: TrainingEngine,
        validation_engine: ValidationEngine,
        visualization_engine: VisualizationEngine,
        test_engine: TestEngine,
        metric_factory: dict[str, partial[Callable]] | None,
        output_dir: str,
        seed: int = 42,
        deterministic: bool = False,
        do_train: bool = True,
        do_validation: bool = True,
        do_visualization: bool = False,
        do_test: bool = True,
        vis_batch_size: int = 64,
    ):
        self._task_type = task_type
        self._data_pipeline = data_pipeline
        self._model_pipeline = model_pipeline
        self._training_engine = training_engine if do_train else None
        self._validation_engine = validation_engine if do_validation else None
        self._visualization_engine = visualization_engine if do_visualization else None
        self._test_engine = test_engine if do_test else None
        self._output_dir = output_dir
        self._seed = seed
        self._deterministic = deterministic
        self._do_train = do_train
        self._do_validation = do_validation
        self._do_visualization = do_visualization
        self._do_test = do_test
        self._visualization_batch_size = vis_batch_size
        self._metric_factory = metric_factory

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def validation_dataloader(self):
        return self._validation_dataloder

    @property
    def test_dataloader(self):
        return self._test_dataloader

    @property
    def visualization_dataloader(self):
        return self._visualization_data_loader

    @property
    def model_pipeline(self):
        return self._model_pipeline

    @property
    def training_engine(self):
        return self._training_engine

    @property
    def validation_engine(self):
        return self._validation_engine

    @property
    def visualization_engine(self):
        return self._visualization_engine

    @property
    def test_engine(self):
        return self._test_engine

    @property
    def device(self):
        return self._device

    @property
    def logger(self):
        return logger

    @property
    def tb_logger(self):
        return self._tb_logger

    def _initialize_runtime(self) -> None:
        import ignite.distributed as idist

        from atria_ml.training.utilities.torch_utils import _initialize_torch

        # initialize training
        _initialize_torch(seed=self._seed, deterministic=self._deterministic)

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

    def _setup_logging(self) -> None:
        import logging

        from atria_ml.training.utilities.torch_utils import _setup_tensorboard

        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(
                    f"Verbose logs can be found at file: {handler.baseFilename}"
                )

        # initialize logging directory and tensorboard logger
        self._tb_logger = _setup_tensorboard(self._output_dir)

    def _build_data_pipeline(self):
        from atria_core.utilities.common import _msg_with_separator

        # build data module
        logger.info(_msg_with_separator("Setting up data pipeline"))
        self._data_pipeline.build()

    def _build_model_pipeline(self) -> None:
        from atria_core.utilities.common import _msg_with_separator

        # initialize the task module from partial
        logger.info(_msg_with_separator("Setting up task module"))
        self._model_pipeline = self._model_pipeline.build(
            dataset_metadata=self._data_pipeline.dataset_metadata,
            tb_logger=self._tb_logger,
        )
        print(self._model_pipeline)

    def _build_dataloaders(self) -> None:
        logger.info("Initializing train dataloader.")
        self._train_dataloader = self._data_pipeline.train_dataloader()
        if self._validation_engine is not None:
            # check if validation dataset is available
            logger.info("Initializing validation dataloader.")
            self._validation_dataloder = self._data_pipeline.validation_dataloader()
            if self._validation_dataloder is None:
                logger.warning(
                    "You have set do_validation=True but there is no validation dataset available. "
                    "To create a validation dataset from the training dataset, set dataset_splitter in the config. "
                    "Using test dataset for validation."
                )
                self._validation_dataloder = self._data_pipeline.test_dataloader()
        if self._visualization_engine is not None:
            # initilize the validation engine from partial
            # by default, visualization engine uses the train dataloader as it is
            # generally to be used for generative tasks
            logger.info("Initializing train dataloader for visualization.")
            self._visualization_data_loader = self._data_pipeline.validation_dataloader(
                batch_size=self._visualization_batch_size
            )
            if self._visualization_data_loader is None:
                self._visualization_data_loader = self._data_pipeline.test_dataloader(
                    batch_size=self._visualization_batch_size
                )
        logger.info("Initializing test dataloader.")
        self._test_dataloader = self._data_pipeline.test_dataloader()

    def _setup_inferencer_config(self) -> None:
        """
        Setup the inferencer config for the training engine.
        This method creates a inferencer config from current trainer config
        and sets the runtime transforms and model pipeline as passed to the trainer for future inference of the model.
        This config is saved in the model checkpoint for future use.
        """
        from hydra_zen import builds

        config = OmegaConf.create(
            builds(
                Inferencer,
                populate_full_signature=True,
                inference_engine=builds(InferenceEngine, populate_full_signature=True),
            )
        )
        config.runtime_transforms = (
            self._run_config._data.data_pipeline.runtime_transforms
        )
        config.model_pipeline = self._run_config._data.model_pipeline
        return InferenceConfig(config)

    def _build_training_engine(self) -> None:
        from atria_core.utilities.common import _msg_with_separator

        if self._validation_engine is not None:
            logger.info(_msg_with_separator("Setting up validation engine"))
            self._validation_engine = self._validation_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=self._validation_dataloder,
                device=self._device,
                metric_factory=self._metric_factory,
                tb_logger=self._tb_logger,
            )

        if self._visualization_engine is not None:
            logger.info(_msg_with_separator("Setting up visualization engine"))
            self._visualization_engine = self._visualization_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=self._visualization_data_loader,
                device=self._device,
                metric_factory=self._metric_factory,
                tb_logger=self._tb_logger,
            )

        # initilize the test engine from partial
        if self._training_engine is not None:
            logger.info(_msg_with_separator("Setting up training engine"))
            self._training_engine = self._training_engine.build(
                run_config=self._run_config,
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=self._train_dataloader,
                device=self._device,
                metric_factory=self._metric_factory,
                tb_logger=self._tb_logger,
                validation_engine=self._validation_engine,
                visualization_engine=self._visualization_engine,
                inference_config=self._setup_inferencer_config(),
            )

    def _build_test_engine(self) -> None:
        from atria_core.utilities.common import _msg_with_separator

        if self._test_engine is not None:
            logger.info(_msg_with_separator("Setting up test engine"))
            self._test_engine = self._test_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=self._test_dataloader,
                device=self._device,
                metric_factory=self._metric_factory,
                tb_logger=self._tb_logger,
            )

    def build(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._initialize_runtime()
        self._setup_logging()
        self._build_data_pipeline()
        self._build_model_pipeline()
        self._build_dataloaders()

    def train(self) -> None:
        if self._do_train:
            self._setup_inferencer_config()
            self._build_training_engine()
            self._training_engine.run()

    def test(self) -> None:
        if self._do_test:
            self._build_test_engine()
            self._test_engine.run()

    def run(self) -> None:
        try:
            self.train()
        except KeyboardInterrupt as e:
            logger.warning(f"Training interrupted: {e}")

        try:
            self.test()
        except KeyboardInterrupt as e:
            logger.warning(f"Testing interrupted: {e}")
