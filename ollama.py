from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama

# Step 1: Convert Image to Caption (using BLIP)
def get_image_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Step 2: Ask Gemma about the caption
def get_gemma_response(caption):
    prompt = f"This is an image description: '{caption}'. Based on this, what can you say about biodiversity or the environment in the image?"
    response = ollama.chat(
        model='gemma:2b',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Main function
if __name__ == "__main__":
    image_path = "coffee_farm.jpg"  # replace with your actual image
    caption = get_image_caption(image_path)
    print("🖼️ Image Caption:", caption)

    answer = get_gemma_response(caption)
    print("🧠 Gemma's Response:\n", answer)
