# LLaMA Factory Florence-2



Download data from https://huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers/tree/main

```bash
wget https://huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers/raw/main/annotations_no_caption.json
```

Convert `annotations_no_caption.json` to LLaMA-Factory format:

```bash
python3 convert.py annotations_no_caption.json TF-ID-arxiv-papers.json
```

Copy `TF-ID-arxiv-papers.json` to LLaMA-Factory/data dir, and add new dataset in `dataset_info.json` file:

```json
 "TF-ID-arxiv-papers": {
    "file_name": "TF-ID-arxiv-papers.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
```

Edit `LLaMA-Factory/examples/train_full/florence2_full_sft.yaml` file, change `dataset` to `TF-ID-arxiv-papers`

Start training:

```bash
llamafactory-cli train examples/train_full/florence2_full_sft.yaml
```

After training finished, modify `saves/florence2-large/full/sft/config.json` file(I still haven't figured out why the saved config has the following issues that need to be modified; it shouldn't be necessary.):

Change `auto_map`, from:

```json
  "auto_map": {
    "AutoConfig": "configuration_florence2.Florence2Config",
    "AutoModel": "modeling_florence2.Florence2ForConditionalGeneration",
    "AutoModelForCausalLM": "microsoft/Florence-2-large-ft--modeling_florence2.Florence2ForConditionalGeneration"
  }
```

to

```json
  "auto_map": {
    "AutoConfig": "configuration_florence2.Florence2Config",
    "AutoModelForCausalLM": "modeling_florence2.Florence2ForConditionalGeneration"
  }
```

Change vision_config's `model_type`

```json
{
  'vision_config': {
      ...
      "model_type": "davit"
      ...
  }
}
```
