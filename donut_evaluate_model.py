"""Predict and evaluate samples."""

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

if __name__ == '__main__':
    model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)

    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")

    question = "What is the invoice number?"
    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

    inputs = processor(image, task_prompt, return_tensors="pt")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        max_length=50
    )
    
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    if question in answer:
        answer = answer.split(question)[-1].strip()
    print(answer)
