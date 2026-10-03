import os
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('digit_recognizer.h5')

def predict_digit(img):
    if img is None:
        return None

    composite = img.get('composite') if isinstance(img, dict) else img

    if composite is not None:
        img_pil = Image.fromarray(composite.astype('uint8')).convert('L')
        arr = np.array(img_pil)
        arr = 255 - arr
    else:
        layers = img.get('layers', [])
        layer = layers[0]
        arr = layer[:, :, 3]

    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = Image.fromarray(arr).crop((x0, y0, x1, y1))
    cropped.thumbnail((20, 20), Image.LANCZOS)

    min_width = 8
    if cropped.width < min_width:
        new_height = cropped.height
        cropped = cropped.resize((min_width, new_height), Image.LANCZOS)

    new_img = Image.new('L', (28, 28), 0)
    upper_left = ((28 - cropped.width) // 2, (28 - cropped.height) // 2)
    new_img.paste(cropped, upper_left)

    processed = np.array(new_img) / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    prediction = model.predict(processed)
    confidences = {str(i): float(prediction[0][i]) for i in range(10)}

    return confidences

demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="🔢 Handwritten Digit Recognizer",
    description="Draw a digit (0-9) below and let the AI guess what it is!",
    theme=gr.themes.Soft(),
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
