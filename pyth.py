from langchain_community.llms import CTransformers
import os

def test_model(model_path, model_type="phi3"):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        llm = CTransformers(
            model=model_path,
            model_type=model_type,
            config={"max_new_tokens": 256, "temperature": 0.7, "context_length": 2048}
        )
        print(f"Successfully loaded model from {model_path}")
        print(llm("Hello, test!"))
        return llm
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

# Test original model
original_path = "/Users/pranav/Documents/mitadt_chatbot/Phi-3-mini-4k-instruct-q4.gguf"
test_model(original_path)

# Test alternative model
alt_path = "/Users/pranav/Documents/mitadt_chatbot/Phi-3-mini-4k-instruct.Q4_0.gguf"
if os.path.exists(alt_path):
    test_model(alt_path)
else:
    print(f"Alternative model not found at {alt_path}")