import hydra
from pathlib import Path
from hydra.utils import get_original_cwd
from mlops_project.data import ArxivPapersDataset

@hydra.main(version_base="1.3", config_path="../../configs", config_name="evaluate_config")
def evaluate():

    test_dataset = ArxivPapersDataset("test", data_dir=Path(f"{get_original_cwd()}/data"))

if __name__ == "__main__":
    evaluate()