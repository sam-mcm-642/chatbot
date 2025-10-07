from mlx_lm import load, generate
import json

def evaluate_model(
    model_path,
    test_prompts_file="data/test_prompts.json"
):
    """Compare base vs fine-tuned model responses"""
    
    # Load models
    print("Loading base model...")
    base_model, base_tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load(model_path)
    
    # Load test prompts
    with open(test_prompts_file) as f:
        prompts = json.load(f)
    
    # Compare responses
    print("\n" + "="*80)
    for i, prompt in enumerate(prompts[:5], 1):  # First 5 prompts
        print(f"\nTest {i}: {prompt}")
        print("-"*80)
        
        print("\nBase model:")
        base_response = generate(base_model, base_tokenizer, prompt=prompt, max_tokens=100)
        print(base_response)
        
        print("\nFine-tuned model:")
        ft_response = generate(ft_model, ft_tokenizer, prompt=prompt, max_tokens=100)
        print(ft_response)
        
        print("="*80)

if __name__ == "__main__":
    evaluate_model("models/fine_tuned/my_assistant")