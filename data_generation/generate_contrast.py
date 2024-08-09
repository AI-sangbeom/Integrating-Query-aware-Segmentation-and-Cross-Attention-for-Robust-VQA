
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
    # 세그먼트된 영역을 0으로 설정하는 대신 원본 이미지의 값을 유지
    ci = segment_output >= 0
    original_image = image_to_tensor(resize_image).expand_as(segment_output).to("cuda")
    # 원본 코드에 명암 조절 추가
    contrast_adjusted_image = adjust_contrast(original_image, factor=0.4)  
    # 세그먼트된 영역은 원본 이미지, 나머지는 명암 조절된 이미지 사용
    results = torch.where(segment_output >= 0, original_image, contrast_adjusted_image)
    segmented_image = to_pil_image(results) 
    segmented_image.save(f'./segment_contrast/{mode}/{image_id}')
    