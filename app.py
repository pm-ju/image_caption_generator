import gradio as gr
from PIL import Image
from model_utils import predict_caption
import json
import httpx
import os # Import the os library to access environment variables

# --- Advanced CSS for a Sleek, Animated UI ---
# (CSS remains the same)
custom_css = """
/* Google Font for a clean, modern aesthetic */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

/* --- Light Mode Styles (Default) --- */
body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.gradio-container {
    max-width: 900px !important;
    margin: 2rem auto !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
}
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: 700;
    color: #1a202c; /* Very dark gray */
    padding-top: 25px;
}
#subtitle {
    text-align: center;
    font-size: 1.1em;
    color: #3b4252; /* Darkened text color */
    margin-bottom: 25px;
}
#output_caption {
    font-size: 1.2em !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    background-color: #f7fafc !important;
    color: #2d3748 !important;
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}
/* Style for the Gradio footer */
.footer {
    color: #a0aec0 !important; /* Muted gray for better visibility */
}

/* --- Dark Mode Specific Styles --- */
/* This targets Gradio's dark mode class for more reliability */
.dark body {
    background: #1a202c; /* Dark blue-gray background */
}
.dark .gradio-container {
    background: #2d3748 !important; /* Darker card background */
    border-color: #4a5568 !important;
}
.dark #title {
    color: #e2e8f0; /* Light gray for title */
}
.dark #subtitle {
    color: #a0aec0; /* Lighter gray for subtitle */
}
.dark #output_caption {
    background-color: #1a202c !important;
    color: #e2e8f0 !important;
    border-color: #4a5568 !important;
}
/* Ensure all labels are visible in dark mode */
.dark .gradio-panel > .label, .dark .gradio-textbox > .label, .dark .gradio-image > .label, .dark .gradio-examples > .label {
    color: #cbd5e0 !important;
}
/* Style for the Gradio footer in dark mode */
.dark .footer {
    color: #718096 !important; /* A slightly lighter gray for dark mode footer */
}


/* --- General Animations (for both themes) --- */
.shimmer .gradio-textbox::placeholder {
    animation: shimmer 2s infinite;
    background: linear-gradient(to right, #e2e8f0 4%, #f7fafc 25%, #e2e8f0 36%);
    background-size: 1000px 100%;
    color: transparent;
}
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}
#output_caption.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.gradio-button.gr-button-primary {
    background: linear-gradient(90deg, #3182ce, #4299e1) !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3) !important;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.03); }
    100% { transform: scale(1); }
}
.gradio-button.gr-button-primary:hover {
    box-shadow: 0 6px 20px rgba(66, 153, 225, 0.5) !important;
    animation: none;
}
"""

# --- Function to call Gemini API with API Key from Environment Variable ---
async def get_stylized_caption(base_caption, style):
    """Calls the Gemini API to rewrite a caption in a specific style."""
    if style == "Descriptive (Default)":
        return base_caption

    # --- MODIFIED: Read API key from environment variables ---
    # This is the standard, secure way to handle API keys.
    # Hugging Face will inject this from "Space secrets".
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found. Please set it in your Hugging Face Space secrets."

    style_prompts = {
        "Humorous": f"Rewrite the following image caption to be funny and witty. You can use cuss words and be honestly funny atleast and write only one response and not give options for your reponse. be creative with cuss words and also use creative cuss words: '{base_caption}'",
        "Poetic": f"Rewrite the following image caption as a short, elegant small poem. It should be elegant and just make people fall in love with it and dont give any options for this atleast. Make it slightly long. You can use cuss words or slangs to give emotions: '{base_caption}'",
        "Dramatic": f"Rewrite the following image caption with a dramatic and epic tone. make something that can make a person cry or make him so emerge in that and make it not too long but enough long and you can use cuss words to give emotions: '{base_caption}'",
        "Philosophical": f"Take this simple image description, '{base_caption}', and rewrite it as a short, thought-provoking philosophical reflection and dont make it too big. You can use some cuss words no restrictions",
        "Historical Context": f"Based on the image description '{base_caption}', invent a plausible and interesting historical context or backstory for the scene. Write it as a short, engaging narrative. dont make it too big, and you can use cuss words"
    }
    user_query = style_prompts.get(style, base_caption)
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": user_query}]}]}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            stylized_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: Could not parse stylized caption.")
            return stylized_text.strip()
        else:
            error_details = response.text
            print(f"API Error Response (Status {response.status_code}): {error_details}")
            if "API_KEY_INVALID" in error_details:
                return "Error: The provided API key is invalid. Please check your Hugging Face Space secrets."
            return f"Error: API returned status {response.status_code}. See logs."

    except Exception as e:
        print(f"API Call Exception: {e}")
        return f"Error: Could not connect to the style generation service. Check network or logs."

# --- Gradio App with Style Selection ---
with gr.Blocks(css=custom_css, theme=gr.themes.Base(primary_hue="blue")) as demo:
    # Title and Subtitle
    gr.Markdown("# ✨ AI Image Caption Pro", elem_id="title")
    gr.Markdown("Upload an image, and our AI will craft a descriptive caption. This model combines a state-of-the-art CNN and LSTM, trained on the MS COCO dataset.", elem_id="subtitle")

    # Main layout with two columns
    with gr.Row(variant="panel"):
        # Left column for image input
        with gr.Column(scale=2):
            image_input = gr.Image(type="numpy", label="Upload Your Image Here")
            style_selector = gr.Dropdown(
                ["Descriptive (Default)", "Humorous", "Poetic", "Dramatic", "Philosophical", "Historical Context"],
                label="Choose a Caption Style",
                value="Descriptive (Default)"
            )
            gr.Examples(
                examples=[["images/dog.jpg"]],
                inputs=image_input,
                label="Click an example to try"
            )

        # Right column for the output and buttons
        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Generated Caption",
                elem_id="output_caption",
                interactive=False,
                placeholder="Your caption will appear here...",
                lines=5
            )
            submit_btn = gr.Button("Generate Caption", variant="primary")
            clear_btn = gr.ClearButton()

    # --- Backend Function with Multi-step UI updates for better UX ---
    async def image_to_caption_ux(image, style):
        if image is None:
            yield "", gr.update(elem_classes=[], placeholder="Your caption will appear here...")
            return

        loading_text = "Generating descriptive caption..."
        yield "", gr.update(placeholder=loading_text, elem_classes=["shimmer"])

        pil_image = Image.fromarray(image)
        base_caption = predict_caption(pil_image)
        
        if style != "Descriptive (Default)":
            loading_text_style = f"Stylizing caption as {style.lower()}..."
            #yield base_caption, gr.update(placeholder=loading_text_style, elem_classes=["shimmer"])
            final_caption = await get_stylized_caption(base_caption, style)
        else:
            final_caption = base_caption

        yield final_caption, gr.update(placeholder="Your caption will appear here...", elem_classes=["fade-in"])

    # --- Connect Components to the Function ---
    submit_btn.click(
        fn=image_to_caption_ux,
        inputs=[image_input, style_selector],
        outputs=[caption_output, caption_output]
    )
    clear_btn.add([image_input, caption_output]).click(
        lambda: ("", gr.update(placeholder="Your caption will appear here...", elem_classes=[])),
        outputs=[caption_output, caption_output]
    )

# Launch the app!
demo.launch()

