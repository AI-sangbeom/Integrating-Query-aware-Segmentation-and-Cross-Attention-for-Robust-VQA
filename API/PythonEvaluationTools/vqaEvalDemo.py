# coding: utf-8

import sys
dataDir = '../'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--m',
    help='input your name of backbone model',
    default='qwen',
    type=str
)
args = parser.parse_known_args()[0]

# set up file names and paths
dataDir='./..'
split = 'val'
annFile='./../../data/Annotations/%s.json'%(split)
imgDir = '%s/Images/' %dataDir

# An example result json file has been provided in './Results' folder.  
resultType  ='fake'
fileTypes   = ['results', 'accuracy', 'captionMetric', 'evalQA', 'evalAnsType', 'answerability'] 
[resFile, accuracyFile, captionMetricFile, evalQAFile, evalAnsTypeFile, answerabilityFile] = ['%s/%s/%s_%s_%s.json'%(dataDir, args.m, split, resultType, fileType) for fileType in fileTypes]  
fdir = os.path.join(dataDir, args.m)
if not os.path.exists(fdir):
	os.mkdir(fdir)

# create vqa object and vqaRes object
vqa = VQA(annFile)
vqaRes = VQA(resFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2


"""
If you have a list of images on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the images in annotation file
"""

# evaluate VQA results
vqaEval.evaluate() 

# print accuracies
print("\n")
print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
print("\n")
print("Per Answer Type Accuracy is the following:")
for ansType in vqaEval.accuracy['perAnswerType']:
	print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
print("\n")

print("Caption metrics are :")
for k, v in list(vqaEval.caption_metric.items()):
	print("%s: %.2f"%(k,v))

# save evaluation results to ./Results folder
json.dump(vqaEval.accuracy,     	open(accuracyFile,     'w'))
json.dump(vqaEval.caption_metric,	open(captionMetricFile,'w'))
json.dump(vqaEval.evalQA,       	open(evalQAFile,       'w'))
json.dump(vqaEval.evalAnsType,  	open(evalAnsTypeFile,  'w'))

# evaluate unanswerability
vqaEval.evaluate_unanswerability()
print("\n\nUnanswerability: ")
print("Average precision: %.2f"%vqaEval.unanswerability['average_precision'])
print("F1 score: %.2f"%vqaEval.unanswerability['f1_score'])
json.dump(vqaEval.unanswerability,  open(answerabilityFile,  'w'))