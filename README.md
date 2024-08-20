# Integrating Query-aware Segmentation and Cross Attention for Robust VQA for CVPR VizWiz Visual Question Answering 

<img src='Image/model.png'>

## Dataset and test code

Download dataset : https://vizwiz.org/tasks-and-datasets/vqa/

You also use test or validation code in this link. 

We use Qwen-VL-Chat as a backbone model. 


## Load model directly
- You can use the Qwen-VL-Chat as follows.
- from transformers import AutoModelForCausalLM
- model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
