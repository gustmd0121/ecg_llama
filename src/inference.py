import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from PIL import Image
from transformers import AutoTokenizer
from model.configuration_llama import MultimodalLlamaConfig
from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from processing_llama import MultiModalLlamaProcessor
from transformers import AutoImageProcessor, AutoTokenizer
from utils.constants import IMAGE_TOKEN, PAD_TOKEN

def load_model(model_path):
    # Load configuration and model
    config = MultimodalLlamaConfig.from_pretrained(model_path)
    model = MultimodalLlamaForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True
    )
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    processor = MultiModalLlamaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer
    )
    
    return model, processor, tokenizer

def generate_response(model, processor, tokenizer, image_path, question):
    # Load and process image
    image = Image.open(image_path)
    image_tensor = processor.image_processor(image, return_tensors="pt")["pixel_values"].cuda()
    
    # Prepare prompt with image token
    prompt = f"Question: {question}\nAnswer: {IMAGE_TOKEN}"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer: ")[-1].strip()

if __name__ == "__main__":
    # Example usage
    model_path = "/home/hschung/ecg-llm/llama-multimodal-vqa/src/model_checkpoints/clip_frozen_image/final_model"
    image_path = "/nfs_edlab/hschung/ecg_plots/records500/10000/10127_hr/10127_hr.jpg"
    question = "Does this ECG display any features of non-diagnostic t abnormalities?"
    
    model, processor, tokenizer = load_model(model_path)
    response = generate_response(model, processor, tokenizer, image_path, question)
    print(f"Question: {question}")
    print(f"Answer: {response}")