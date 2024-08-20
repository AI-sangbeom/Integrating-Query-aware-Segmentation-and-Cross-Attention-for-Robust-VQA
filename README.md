## CVPR VizWiz Visual Question Answering 
# Integrating Query-aware Segmentation and Cross Attention for Robust VQA 

<p align="center">
  <img src='Image/model.png'>
</p>


## Dataset and test code
  - Download dataset : https://vizwiz.org/tasks-and-datasets/vqa/
  - You also download test/validation code at this link. 

## Load model directly
  - We use Qwen-VL-Chat as a backbone model. 
  - You can use the Qwen-VL-Chat as follows.
<pre>
<code>
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
</code>
</pre>
  - Then, replace files with Qwen-VL-Chat_core
