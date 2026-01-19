import hydra
from omegaconf import DictConfig, OmegaConf
from importlib import import_module
from hydra.utils import instantiate

def load_callable(path: str):
    """Convert string path to actual callable."""
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)

@hydra.main(version_base=None, config_path="configs", config_name="test_dataset")
def main(cfg: DictConfig):
    print("=== Hydra Config ===")
    print(OmegaConf.to_yaml(cfg))

    # Convert loader_func string to callable locally
    loader_func = load_callable(cfg.dataset.loader_func)
    dataset_cfg = dict(cfg.dataset)
    dataset_cfg['loader_func'] = loader_func

    # Instantiate dataset using Hydra
    dataset = instantiate(dataset_cfg)
    dataset.load()
    dataset.summary()

    # Preprocess features
    preprocessor = instantiate(cfg.preprocessing)
    X_processed = preprocessor.fit_transform(dataset.data)

    print("\n=== Processed Features ===")
    print(X_processed.head())

    # Target
    y = dataset.target
    print("\n=== Target ===")
    print(y.head())

if __name__ == "__main__":
    main()
