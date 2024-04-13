import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

# login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "/bask/homes/a/asiw9691/PathVLM/UNI"
# os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
# hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.Resize(256), # 先将最短边调整到256像素
        transforms.CenterCrop(224), # 然后从中心裁剪出224x224像素的图像
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ]
)
model.eval()

from PIL import Image
image = Image.open("/bask/homes/a/asiw9691/PathVLM/source/Flamingo/med-flamingo/img/test_path5.jpg")

image = transform(image) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
with torch.inference_mode():
    feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,1024]

    print(feature_emb.shape)
