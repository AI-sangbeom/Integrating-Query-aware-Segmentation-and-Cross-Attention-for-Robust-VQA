
import os 
import json 
import torch 

from tqdm import tqdm 
from PIL import Image 
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation 

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda")

data_dir = "/abr/wonjun/cvpr_chal/vizwiz"
mode = "train"

def adjust_contrast(image, factor):
    mean = image.mean([1, 2], keepdim=True)
    return torch.clamp((image - mean) * factor + mean, 0, 1)

with open(f'{data_dir}/{mode}.json','r')as f : 
    data = json.load(f)
    
image_to_tensor = ToTensor()

for i in tqdm(range(len(data))) : 
    image_id = data[i]['image'] ; print(image_id)
    image = Image.open(os.path.join(data_dir,mode,image_id))
    resize_ = transforms.Resize([352,352])
    resize_image = resize_(image)
    question = [data[i]['question']]
    inputs = processor(text=question, images=[resize_image] * len(question), padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(0)
    predict_value = preds.expand(3,352,352)
    mean = predict_value.mean([1,2 ], keepdim=True)
    std = predict_value.std([1,2 ], keepdim=True)
    segment_output = (predict_value - mean) / (std + 1e-5) 
    segment_output = torch.tanh(segment_output).to("cuda")

    # ver 2 
    # 원본 이미지에서 세그먼트된 부분의 값을 가져오도록 수정
    results = torch.where(ci, original_image, segment_output)
    segmented_image = to_pil_image(results)

    ci = segment_output >= 0
    segment_output[ci] = 0
    original_image = image_to_tensor(resize_image).to("cuda")
    results = segment_output + original_image
    segmented_image = to_pil_image(results)
    segmented_image.save(f'./segment_highlight/{mode}/{image_id}')
