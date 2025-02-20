import sys
from transformers import pipeline, AutoProcessor
from PIL import Image
import requests

def main():
    # Define the model ID
    model_id = "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"
    
    # Initialize the image-to-text pipeline
    try:
        pipe = pipeline("image-to-text", model=model_id, device='cuda:0')
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)
    
    # URL of the image to be used
    image_path = './datasets/PLD/Images/Bell Pepper___Bacterial_spot/image (1).JPG'
    # Load the image

    try:
        #image = Image.open(requests.get(image_url, stream=True).raw)
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading the image: {e}")
        sys.exit(1)
    
    # Initialize the processor
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading the processor: {e}")
        sys.exit(1)
    
    # Initialize conversation history
    conversation = []
    
    print("=== LLaVA Multi-Round Chat ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        try:
            # Prompt user for input
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting the conversation.")
            break
        
        # Exit condition
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the conversation.")
            break
        
        # Append user message to conversation history
        if not conversation:
            # First message includes the image
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image"},
                ],
            })
        else:
            # Subsequent messages are text-only
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                ],
            })
        
        # Generate the prompt using the conversation history
        try:
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception as e:
            print(f"Error generating prompt: {e}")
            continue
        
        # Generate the model's response
        try:
            outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
            response_text = outputs[0].get("generated_text", "").strip().split('ASSISTANT: ')[-1]
        except Exception as e:
            print(f"Error during generation: {e}")
            continue
        
        # Display the assistant's response
        print(f"Assistant: {response_text}\n")
        
        # Append assistant's response to conversation history
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": response_text},
            ],
        })

if __name__ == "__main__":
    main()
