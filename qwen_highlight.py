import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import math

from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, tqdm_notebook

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from EarlyStopping import EarlyStopping
import os, json, torch, math
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset


import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--world_size", type=int, required=True, default=1,
                              help="select number: how many gpu use")
parser.add_argument("--batch_size", type=int, required=True, default=1,
                              help="select number: how many batch use")
parser.add_argument("--epochs", type=int, required=True, default=1,
                              help="select number: how many epoch use")

parser.add_argument("--train_path_1", type=str, required=False, default=1,
                              help="TL_tableqa_라벨링데이터.json")
# parser.add_argument("--valid_path_1", type=str, required=True, default=1,
#                               help="VL_tableqa_라벨링데이터.json")

parser.add_argument("--save_directory", type=str, required=True, default=1,
                              help="./sy_checkpoint_original2")

parser.add_argument("--port", type=int, required=True, default=1,
                              help="9992")

answer_class = ["unanswerable", "other", "number", "yes", "no"]
args = parser.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen_VL_Chat_sy", trust_remote_code="cpu")
# print(tokenizer.eod_id)
tokenizer.pad_id = tokenizer.eod_id
# tokenizer.pad_token = tokenizer.eod_id
tokenizer.pad_token_id = tokenizer.eod_id
import os
import json
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


class Vizwiz(Dataset):
    def __init__(self, tokenizer, max_len=400):
        data_path = "/abr/wonjun/cvpr_chal/vizwiz/train.json"
        with open(data_path, 'r+') as f:
            self.csv_file = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = 500

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        from transformers.trainer_pt_utils import LabelSmoother
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index

        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        system_message: str = "You are a helpful assistant."
        im_start = self.tokenizer.im_start_id
        im_end = self.tokenizer.im_end_id
        nl_tokens = self.tokenizer('\n').input_ids
        _system = self.tokenizer('system').input_ids + nl_tokens
        _user = self.tokenizer('user').input_ids + nl_tokens
        _assistant = self.tokenizer('assistant').input_ids + nl_tokens
        
        temp_name = self.csv_file[idx]['image'].split("_")[0]
        if temp_name == "COCO":
            image = "/raid/wonjun/Qwen-VL/vqav2_gen/segment_image_vqa/train/" + self.csv_file[idx]['image']
        elif temp_name == "VizWiz":
            image = "/raid/wonjun/Qwen-VL/segment_highlight/train/" + self.csv_file[idx]['image']
            
        question = self.csv_file[idx]['question']
        answers  = self.csv_file[idx]['answer']
        max_len = self.max_len
        
        sources = [[
            {
                'from': 'user',
                'value': "<img>"+image + "</img>\n" + """The correct answer type is one of ["number", "words", "yes", "no"]. If it is impossible to answer an image-related question or there is no existing information, please reply as "unanswerable". Question:""" +question
            },
            {
                'from': "assistant",
                'value': answers
            },
        ]]
        
        # Apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != roles["user"]:
                source = source[1:]

            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target)
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        input_ids = torch.tensor(input_ids)
        targets = torch.tensor(targets)[0].tolist()
        attention_mask = input_ids.ne(tokenizer.pad_token_id)[0].tolist()
        attention_mask = [1 if item else 0 for item in attention_mask]

        answerable = int(self.csv_file[idx]['answerable'])
        if answers == "unanswerable":
            answer_type = "unanswerable"
            answerable = 0
        elif answers.lower() == "yes":
            answer_type = "yes"
        elif answers.lower() == "no":
            answer_type = "no"
        elif self.csv_file[idx]['answer_type'] == 'yes/no':
            None
        else:
            answer_type = self.csv_file[idx]['answer_type']

        return torch.LongTensor(input_ids[0].tolist()), torch.LongTensor(attention_mask), torch.LongTensor(targets), question, answers, answerable, answer_type
    
def find_last_continuous_sequence_index(my_list, target_values):
    last_index = len(my_list) - 1
    for i, value in enumerate(reversed(my_list)):
        if value == target_values[0]:
            if i + 1 < len(target_values) and my_list[last_index - i - 1] == target_values[i + 1]:
                continue
            else:
                return last_index - i
    return -1  # 연속된 값이 없을 경우 -1을 반환

def get_dataset():
    world_size = dist.get_world_size()
    
    import torch.utils.data as torchdata
    from torch.utils.data import DataLoader
    # from dataloader_vizwiz_seg_last import Vizwiz
    
    print("====== Data Loader ======")
    # train_loaded = CustomDataset(glob_files=glob_files)
    train_loaded = Vizwiz(tokenizer)#, transform=transform_train)
    train_sampler = DistributedSampler(train_loaded, num_replicas=world_size)
    batch_size = int(args.batch_size)
    print(world_size, batch_size)
    
    train_loader = DataLoader(
        dataset=train_loaded,
        sampler=train_sampler,
        batch_size=batch_size
    )
    
    return train_loader, batch_size
    
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        
        
def reduce_dict(input_dict, average=True):
    world_size = float(dist.get_world_size())
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def run(rank, world_size):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")
    # torch.manual_seed(123)
    train_loader, batch_size = get_dataset()
    
    #BERT 모델, Vocabulary 불러오기
        
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    model_path = "Qwen_VL_Chat"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    
    model.generation_config.chat_format = "raw"
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters: ", total)

    for name, params in model.named_parameters():
        if 'lora' in name:
            params.requires_grad = True
        elif "attn_pool" in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("LoRA+Cross Parameters: ", total)


    
    model = DDP(model,device_ids=[rank],output_device=rank)
#     optimizer = AdamW(model.parameters(),
#                   lr = 3e-5, 
#                   eps = 1e-8 

    

    m = lambda x : {key:x[key].to(device) for key in x.keys()}
    ms = lambda x : {key:x[key].to(device) for key in x.keys()}
    # input_ids = m(tokenizer(query, return_tensors="pt", padding=True))

    from torch.nn import CrossEntropyLoss
    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    history =  {
            "rank": rank,
            "train_loss_val": [],
            "train_acc_val": [],
            "val_loss_val": [],
            "val_acc_val": []
        }
    if rank == 0:
        history = {
            "rank": rank,
            "train_loss_val": [],
            "train_acc_val": [],
            "val_loss_val": [],
            "val_acc_val": []
        }
        
        
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []
        
    patience = 20000
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=args.save_directory)
    only_check_device = torch.device("cuda:0")
    print_iter_ = 10
    save_iter_ = 500
    check_once = 0
    for epoch in range(args.epochs):
        for phase in ['Train']:
            start = time.time()
            iter_count = 0
            if phase == "Train":
                
                device = torch.device(f"cuda:{dist.get_rank()}")
                train_num_batches = int(math.ceil(len(train_loader.dataset) / int(args.world_size)))
                train_iter_size = math.ceil(train_num_batches / batch_size)
                model.train()
                # let all processes sync up before starting with a new epoch of training
                # dist.barrier()
                train_loss = 0.0
                total_count = 0
                acc = 0

                # Tracking variables 
                train_accuracy = 0
                temp_train_count = 0
                temp_temp_time = time.time()
                for input_ids, attention_mask, targets, question, answers, answerable, answer_type in train_loader:
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=targets.to(device))
                    loss = outputs.loss
                    
                    average_gradients(model)
                    loss.backward()
                    optimizer.step()
                    loss_ = {'loss': torch.tensor(loss.item()).to(device)}
                    train_loss += loss.item()
                    temp_train_count += 1

                    if device == only_check_device:
                        if iter_count % print_iter_ == 0:
                            print("============================")

                            temp_acc = 0
                            for j in range(len(answer_type)):
                                my_list = input_ids[j].tolist()  # 리스트에는 많은 요소가 있으므로 생략했습니다.
                                target_values = [151644, 77091]
                                last_continuous_sequence_index = find_last_continuous_sequence_index(my_list, target_values)
                                temp_input = torch.LongTensor(my_list[:last_continuous_sequence_index+2]).unsqueeze(0)
                                generated_output = model.module.generate(input_ids=temp_input.to(device), 
                                                                          # do_sample=True,
                                                                          top_p=0.5,
                                                                          top_k=0,
                                                                          # early_stopping=True,
                                                                          max_new_tokens=400,
                                                                        )
                                
                                # response, history_chat = model.module.chat(tokenizer, query= """ The correct answer type is one of ["number", "words", "yes", "no"]. If it is impossible to answer an image-related question or there is no existing information, please reply as "unanswerable". Question: """ + query[j]  +" Answer:", history=None)
                                print("--------------------------")
                                print("Question:", question[j])
                                print("Real:", answers[j])
                                predicted_output = tokenizer.decode(generated_output[0][last_continuous_sequence_index+2:], skip_special_tokens=True).strip()
                                print("Predicted:", predicted_output.strip())


                                # print("training Answer:", tokenizer.decode(results.argmax(2)[j], skip_special_tokens=True))
                                # print("Revise_query:", revise_query[0])
                                print("--------------------------")
                                print("\n")
                                if predicted_output.lower().strip() == answers[j].lower().strip():
                                    acc += 1
                                    temp_acc += 1
                                    total_count += 1
                                else:
                                    total_count += 1
                            print("Batch Accuracy: {:.4f}".format( temp_acc/batch_size))
                            print("Total Accuracy: {:.4f}".format( acc / total_count) )
                            print("============================")
                            print("Now Iteration:", temp_train_count, "/", train_iter_size)
                            print("Loss:", loss, "\n")
                            print("Total_loss:", train_loss/temp_train_count)
                            print("============================")
                            print(f"{time.time()-start:.4f} sec")
                            print(f"{time.time()-temp_temp_time:.4f} sec")
                            temp_temp_time = time.time()
                            print(time.strftime('%Y.%m.%d - %H:%M:%S')) # 년.월.일 - 시간
                
                    cleanup
                    iter_count += 1
                    
                    
                    if rank == 0:
                        if iter_count % save_iter_ == 0:
                            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
                            # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
                            early_stopping(train_loss / iter_count, model, epoch, iter_count)
                            print(f'Rank {rank} epoch {epoch} iters {iter_count} train_loss {(train_loss/iter_count):.4f}')
                            history['train_loss_val'].append(train_loss / iter_count)
                            # history['val_loss_val'].append(val_loss_val)
                            print(f"{time.time()-start:.4f} sec")

                        if early_stopping.early_stop:
                            print("Early stopping")
                            break
    
                    
                train_loss_val = train_loss / train_iter_size
                print(train_loss_val)
                avg_train_losses.append(train_loss_val)
                print(f"{time.time()-start:.4f} sec")
                
        if rank == 0:
            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
            # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
            early_stopping(train_loss_val, model, epoch, iter_count)
            print(f'Rank {rank} epoch {epoch} iters {iter_count} train_loss {train_loss_val:.4f}')
            # print(f'Rank {rank} epoch {epoch} train_loss {train_loss_val:.4f} valid_loss {val_loss_val:.4f}')
            history['train_loss_val'].append(train_loss_val)
            # history['val_loss_val'].append(val_loss_val)
            print(f"{time.time()-start:.4f} sec")
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print(f'Rank {rank} finished training')
    print(history)
    cleanup(rank)  

def cleanup(rank):
    # dist.cleanup()  
    dist.destroy_process_group()
    print(f"Rank {rank} is done.")
    
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def init_process(
        rank, # rank of the process
        world_size, # number of workers
        fn, # function to be run
        # backend='gloo',# good for single node
        backend='nccl' # the best for CUDA
    ):
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:' + str(args.port),
                            world_size=world_size, 
                            rank=rank)
    dist.barrier()
    setup_for_distributed(rank == 0)
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = args.world_size
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

        
        