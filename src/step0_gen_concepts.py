import json
import openai
import argparse

class OpenAIClient:
    def __init__(self, config, api_key):
        self.config = config
        self.client = openai.OpenAI(api_key=api_key)

    def query_visual_features(self, item_name):
        """Query GPT-4o for visual features in bullet-point format."""
        response = self.client.chat.completions.create(
            model=self.config.args.model, #"gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an AI specialized in providing accurate and detailed descriptions of visual features for various objects."},
                {"role": "user", "content": f"List distinctive visual features of the {item_name} comprehensively, in a bullet-point format without any additional text."}
            ]
        )
        return response.choices[0].message.content

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Query GPT-4o for visual feature extraction.")
        parser.add_argument("--model", type=str, default='gpt-4o-2024-08-06', help="Name of the model to use.") #gpt-4o-2024-08-06,gpt-4o-mini-2024-07-18
        parser.add_argument("--dataset", type=str, default="fgvc", help="Dataset name.")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
        parser.add_argument("--data_path", type=str, default="./datasets/finer", help="Path to the dataset.")
        parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size.")
        parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
        self.args = parser.parse_args()

def load_json(file_path):
    """Load a JSON file from the specified path."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_labels(data):
    """Extract unique labels from the dataset."""
    fine_labels = set()
    for item in data:
        #print("item", item)
        fine_labels.add(item["label"])
    return fine_labels

def extract_visual_concepts(response_text):
    """Extract visual features from GPT-4o response."""
    lines = response_text.splitlines()
    concepts = [line.strip('- ').strip() for line in lines if line.strip() and line.lstrip().startswith('-')]
    return concepts

def main():
    # Initialize configuration
    config = Config()
    #config.args.api_key = api_key
    
    # Define paths
    output_path = f"{config.args.data_path}/{config.args.dataset}/{config.args.dataset}_visual_features.json"
    train_file = f"{config.args.data_path}/{config.args.dataset}/{config.args.dataset}-train.json"

    # Load training data
    train_data = load_json(train_file)

    # Extract unique labels
    fine_labels = extract_labels(train_data)
    #print(fine_labels)

    # Initialize OpenAI client
    client = OpenAIClient(config, api_key=config.args.api_key)

    # Dictionary to store results
    results = {}
    
    # Query visual features for each label and store the results
    for item_id, label in enumerate(fine_labels, start=1):
        print(f"Querying visual features for {label}...")
        raw_response = client.query_visual_features(label)
        print(f"Visual features for {label}:\n\n{raw_response}")
        visual_concepts = extract_visual_concepts(raw_response)
        
        results[item_id] = {
            "name": label,
            "visual_features": visual_concepts
        }
    
    # Save the results to a JSON file
    with open(output_path, 'w') as result_file:
        json.dump(results, result_file, indent=4)
    
    print(f"Querying completed and results saved to '{output_path}'.")

if __name__ == "__main__":
    main()
