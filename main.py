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
    
    chunk_size = yaml_data["chunk_size"]
    chunk_overlap = yaml_data["chunk_overlap"]
    preprocessor = Preprocessor(chunk_size, chunk_overlap)
    texts = loaded_text_list(yaml_data, preprocessor)
    chunks = preprocessor.data_splitter(texts)

    retriever = Retriever()
    retriever.build_index(chunks)

    query = input("Enter your question: ")
    retrieved_chunks = retriever.retrieve(query)

    model = OllamaModel()
    context = "\n".join(retrieved_chunks)
    prompt = f"Use the context below to answer the question:\n\n{context}\n\nQuestion: {query}"
    answer = model.generate(prompt)
    print("💬 Llama3.1 answer:\n", answer)

def loaded_text_list(yaml_data, preprocessor): 
    """ checks if a repo was provided or a
    list of urls and return texts list """

    has_urls = "urls" in yaml_data and yaml_data["urls"]
    has_repo = "repo_url" in yaml_data and yaml_data["repo_url"]

    if has_urls and has_repo:
        raise ValueError("Only provide either `urls` or `repo_url`")
    elif has_urls:
        return preprocessor.data_loader_urls(yaml_data["urls"])
    elif has_repo:
        return preprocessor.data_loader_repo(yaml_data["repo_url"])
    else:
        raise ValueError("Provide either `urls` or `repo_url`")
        
        

if __name__ == "__main__":
    main()
