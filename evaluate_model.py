"""Predict and evaluate samples."""

from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")
model = Pix2StructForConditionalGeneration.from_pretrained(
    "google/pix2struct-docvqa-base",
)

image = Image.open("data/results/rescaled/optimized.png")
question = "How much is the total?"
inputs = processor(images=image, text=question, return_tensors="pt")

# autoregressive generation
generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)  # noqa: T201
