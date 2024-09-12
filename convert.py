import json
from collections import defaultdict

CLASS_MAP = {
    1: "table",
    2: "figure",
}
path = "annotations_no_caption.json"

images_data = {}

with open(path, "r", encoding="utf-8") as fr:
    data = json.load(fr)

for it in data["images"]:
    images_data[it["id"]] = it

annotations_group_by_image_id = defaultdict(list)
for it in data["annotations"]:
    image_id = it["image_id"]
    if image_id not in images_data:
        continue

    annotations_group_by_image_id[image_id].append(it)

outputs = []
for image_id, annotations in annotations_group_by_image_id.items():
    # sort
    annotations = sorted(annotations, key=lambda x: x["category_id"])

    prompt = "<OD>"
    response = ""
    for anno in annotations:
        x, y, w, h = anno["bbox"]
        # normalize to [0, 1000]
        x1 = int(x / images_data[image_id]["width"] * 1000)
        y1 = int(y / images_data[image_id]["height"] * 1000)
        x2 = int((x + w) / images_data[image_id]["width"] * 1000)
        y2 = int((y + h) / images_data[image_id]["height"] * 1000)
        response += (
            f"{CLASS_MAP[anno['category_id']]}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
        )
    outputs.append(
        {
            "messages": [
                {"content": prompt, "role": "user"},
                {"content": response, "role": "assistant"},
            ],
            "images": ["TF-ID-arxiv-papers/" + images_data[image_id]["file_name"]],
        }
    )

with open("TF-ID-arxiv-papers.json", "w", encoding="utf-8") as fw:
    json.dump(outputs, fw, indent=4, ensure_ascii=False)
