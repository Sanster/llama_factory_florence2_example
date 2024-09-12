import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


device = "cuda"
torch_dtype = torch.float16
task = "<OD>"
save_path = "output.png"

model_path = sys.argv[1]
img_p = sys.argv[2]

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch_dtype, trust_remote_code=True
).to(device)

image = Image.open(img_p).convert("RGB")
inputs = processor(text=task, images=image, return_tensors="pt").to(device, torch_dtype)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(generated_text)

parsed_answer = processor.post_process_generation(
    generated_text, task=task, image_size=(image.width, image.height)
)


# Get the size of the original image
dpi = 72  # You can adjust this value if needed
figsize = (image.width / dpi, image.height / dpi)

# Create a figure with the same size as the original image
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

ax.imshow(image)

for bbox, label in zip(parsed_answer[task]["bboxes"], parsed_answer[task]["labels"]):
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox
    # Create a Rectangle patch
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
    )
    # Add the rectangle to the Axes
    ax.add_patch(rect)
    # Annotate the label
    plt.text(
        x1,
        y1,
        label,
        color="white",
        fontsize=8,
        bbox=dict(facecolor="red", alpha=0.5),
    )

ax.axis("off")

# When saving, specify the DPI and remove any padding
plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
print(f"Image saved to {save_path}")
