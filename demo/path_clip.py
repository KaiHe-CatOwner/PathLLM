import torch
from PIL import Image

import open_clip

## load the model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/load_weights/pathclip/pathclip-base.pt',
                                                        force_quick_gelu=True)
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model = model.cuda()

##load the image and prepare the text prompt
img_path = '/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/data/test_data/test_path1.jpg'
label_description_list = ['apple',  'liver', 'cancer',] # specify the label descriptions
text_label_list = ['An image of {}'.format(i) for i in label_description_list]
image = Image.open(img_path)
image = preprocess(image).unsqueeze(0).cuda()
text = tokenizer(text_label_list).cuda()

## extract the img and text feature and predict the label
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    predict_label = torch.argmax(text_probs).item()
    print(predict_label)
