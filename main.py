from data.data_module import DriveDataModule
from data.utils import DriveData
import yaml


def main():
    data_module = DriveDataModule(config["data"]["root"])

if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "rb"), yaml.FullLoader)

    main()