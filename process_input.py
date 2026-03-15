from src import Preprocessor
from src.retriever import Retriever
from src.model_interface import OllamaModel
import argparse
import yaml
 
def main():

    #loading data from configuration file
    parser = argparse.ArgumentParser(description='Generate md with all plots created')
    parser.add_argument(
        "--config",
        required = True,
        dest="config_file",
        default="config.yaml",
        help="Pass config file [default: %default]",
        metavar="STRING",
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print("Argument Error")
        parser.print_help()
        exit()

    config_file = args.config_file
    print(f"Opening config file: {config_file}")
    with open(config_file) as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    preprocessor = Preprocessor(
                    yaml_data["chunk_size"],
                    yaml_data["chunk_overlap"]
                )
    texts = preprocessor.loaded_list(yaml_data)
    chunks = preprocessor.data_splitter(texts)

    retriever = Retriever()
    retriever.build_index(chunks)
    retriever.save("index_store")          # <-- persist to disk
    print("Index built and saved.")

if __name__ == "__main__":
    main()
