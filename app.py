import os
import gc
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Load the trained model once when the app starts
model = load_model("cnn_model.h5")


def predict_digit(img):
    if img is None:
        return None

    # Handle Gradio Sketchpad output
    if isinstance(img, dict):
        composite = img.get("composite")

        if composite is not None:
            img_pil = Image.fromarray(
                composite.astype("uint8")
            ).convert("L")

            arr = np.array(img_pil)

            # Convert black drawing on white background
            # to white digit on black background
            arr = 255 - arr

        else:
            layers = img.get("layers", [])

            if not layers:
                return None

            layer = layers[0]
            arr = layer[:, :, 3]

    else:
        img_pil = Image.fromarray(
            img.astype("uint8")
        ).convert("L")

        arr = np.array(img_pil)
        arr = 255 - arr

    # Find the bounding box of the digit
    coords = np.argwhere(arr > 20)

    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = Image.fromarray(arr).crop(
        (x0, y0, x1, y1)
    )

    # Resize while maintaining aspect ratio
    cropped.thumbnail((20, 20), Image.LANCZOS)

    # Prevent extremely thin digits from becoming too narrow
    min_width = 8

    if cropped.width < min_width:
        new_height = cropped.height

        cropped = cropped.resize(
            (min_width, new_height),
            Image.LANCZOS
        )

    # Create MNIST-style 28x28 image
    new_img = Image.new("L", (28, 28), 0)

    upper_left = (
        (28 - cropped.width) // 2,
        (28 - cropped.height) // 2
    )

    new_img.paste(cropped, upper_left)

    # Convert to float32 and normalize
    processed = np.array(
        new_img,
        dtype=np.float32
    ) / 255.0

    processed = processed.reshape(
        1, 28, 28, 1
    )

    # Run inference directly
    # This uses the exact same trained model and weights
    prediction = model(
        processed,
        training=False
    ).numpy()

    confidences = {
        str(i): float(prediction[0][i])
        for i in range(10)
    }

    # Release temporary Python objects
    del processed
    del prediction
    gc.collect()

    return confidences


# Create the Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="🔢 Handwritten Digit Recognizer",
    description="Draw a digit (0-9) below and let the AI guess what it is!",
    theme=gr.themes.Soft(),
)


# Start the application
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
