import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    logging.info(
        f"Training script. The outputs of hydra will be stored in: {output_path}"
    )
    logging.info("Initializing logger, callbacks and trainer")

    if cfg.machine.name == "slurm":
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg.machine.trainer.devices = num_gpus
        cfg.machine.trainer.num_nodes = num_nodes
        logging.info(f"Slurm config: {num_gpus} gpus,  {num_nodes} nodes")
    trainer = instantiate(cfg.machine.trainer)

    default_ref_dataloader_config = cfg.data.reference_dataloader
    default_query_dataloader_config = cfg.data.query_dataloader

    query_dataloader_config = default_query_dataloader_config.copy()
    ref_dataloader_config = default_ref_dataloader_config.copy()

    if cfg.dataset_name in ["hb", "tless"]:
        query_dataloader_config.split = "test_primesense"
    else:
        query_dataloader_config.split = "test"
    query_dataloader_config.root_dir += f"{cfg.dataset_name}"
    query_dataset = instantiate(query_dataloader_config)

    logging.info("Initializing model")
    model = instantiate(cfg.model)

    model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names
    model.dataset_name = cfg.dataset_name

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,  # only support a single image for now
        num_workers=cfg.machine.num_workers,
        shuffle=False,
    )
    if cfg.model.onboarding_config.rendering_type == "pyrender":
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "pbr":
        logging.info("Using BlenderProc for reference images")
        ref_dataloader_config._target_ = "provider.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = f"{query_dataloader_config.root_dir}"
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        if not os.path.exists(ref_dataloader_config.template_dir):
            os.makedirs(ref_dataloader_config.template_dir)
        ref_dataset = instantiate(ref_dataloader_config)
        ref_dataset.load_processed_metaData(reset_metaData=True)
    else:
        raise NotImplementedError
    model.ref_dataset = ref_dataset

    segmentation_name = cfg.model.segmentor_model._target_.split(".")[-1]
    agg_function = cfg.model.matching_config.aggregation_function
    rendering_type = cfg.model.onboarding_config.rendering_type
    level_template = cfg.model.onboarding_config.level_templates
    model.name_prediction_file = f"result_{cfg.dataset_name}"
    logging.info(f"Loading dataloader for {cfg.dataset_name} done!")
    trainer.test(
        model,
        dataloaders=query_dataloader,
    )
    logging.info(f"---" * 20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
