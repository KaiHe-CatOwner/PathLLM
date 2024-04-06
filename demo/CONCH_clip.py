from conch.open_clip_custom import create_model_from_pretrained
import torch
from PIL import Image

model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/load_weights/conch/pytorch_model.bin")

image = Image.open("/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/data/test_data/test_path1.jpg")
image = preprocess(image).unsqueeze(0)
with torch.inference_mode():
    image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
    print(image_embs.shape)