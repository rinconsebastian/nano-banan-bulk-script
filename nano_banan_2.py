import argparse
import os
import glob
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    HAS_REQUIRED_PACKAGES = True
except ImportError:
    HAS_REQUIRED_PACKAGES = False

# --- CONFIGURATION ---
# Replace 'YOUR_API_KEY_HERE' with your actual Gemini API Key from Google AI Studio.
API_KEY = "you google api key here"
# ---------------------

def process_image(client, model_name, image_path, prompt, dest_dir):
    """Processes a single image using the GenAI model and saves the output."""
    try:
        # Load the image using PIL
        img = Image.open(image_path)
        
        # Call the GenAI model
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, img]
        )
        
        # Save the result
        base_name = Path(image_path).stem
        saved_something = False
        
        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            for i, part in enumerate(parts):
                # Check for image data
                if getattr(part, 'inline_data', None) is not None:
                    mime_type = part.inline_data.mime_type
                    ext = ".png" if "png" in mime_type else ".jpg"
                    if "webp" in mime_type: ext = ".webp"
                    
                    suffix = f"_{i}" if len(parts) > 1 else ""
                    dest_filename = os.path.join(dest_dir, f"{base_name}{suffix}{ext}")
                    
                    with open(dest_filename, "wb") as f:
                        f.write(part.inline_data.data)
                    print(f"✅ Success (Image): {os.path.basename(image_path)} -> {os.path.basename(dest_filename)}")
                    saved_something = True
                
                # Check for text data
                elif getattr(part, 'text', None) is not None:
                    suffix = f"_{i}" if len(parts) > 1 else ""
                    dest_filename = os.path.join(dest_dir, f"{base_name}{suffix}.txt")
                    with open(dest_filename, "w", encoding="utf-8") as f:
                        f.write(part.text)
                    print(f"✅ Success (Text): {os.path.basename(image_path)} -> {os.path.basename(dest_filename)}")
                    saved_something = True
        
        # Fallback if no specific parts found but response.text works
        if not saved_something and getattr(response, 'text', None):
            dest_filename = os.path.join(dest_dir, f"{base_name}.txt")
            with open(dest_filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"✅ Success (Text fallback): {os.path.basename(image_path)} -> {os.path.basename(dest_filename)}")
            saved_something = True
            
        if not saved_something:
            print(f"⚠️ Warning: No supported output content found for {os.path.basename(image_path)}")
            
        return True
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Bulk Image Processing using Nano Banan 2 (Gemini)")
    parser.add_argument("-s", "--source", required=True, help="Path to the source folder containing images")
    parser.add_argument("-d", "--destination", required=True, help="Path to the destination folder for results")
    parser.add_argument("-p", "--prompt", required=True, help="The prompt to apply to all images")
    parser.add_argument("-m", "--model", default="gemini-3.1-flash-image-preview", help="The model to use (default: gemini-3.1-flash-image-preview)")
    
    args = parser.parse_args()

    if not HAS_REQUIRED_PACKAGES:
        print("❌ Missing required packages. Please install them using:")
        print("pip install google-genai pillow")
        return

    # Validate source
    if not os.path.isdir(args.source):
        print(f"❌ Error: Source folder '{args.source}' does not exist.")
        return

    # Create destination if it doesn't exist
    os.makedirs(args.destination, exist_ok=True)

    # Initialize client
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print("❌ Error initializing the GenAI client.")
        print("Ensure you have set a valid API_KEY in this script.")
        print(f"Details: {e}")
        return

    # Find images
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif')
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(args.source, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.source, f"*{ext.upper()}")))

    if not image_paths:
        print(f"⚠️ No images found in '{args.source}'. Supported formats: {', '.join(supported_extensions)}")
        return

    print(f"🍌 Nano Banan 2 Initialized!")
    print(f"Found {len(image_paths)} images to process.")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    success_count = 0
    for idx, path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing {os.path.basename(path)}...")
        if process_image(client, args.model, path, args.prompt, args.destination):
            success_count += 1

    print("-" * 40)
    print(f"🎉 Done! Successfully processed {success_count} out of {len(image_paths)} images.")

if __name__ == "__main__":
    main()
