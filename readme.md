<h1 align="center" style="color:#1976D2; font-size:42px; font-weight:bold; margin-bottom:0;">
 🧭 ComPASS
</h1>

<p align="center" style="color:#42A5F5; font-size:16px; margin-top:0;">
  Towards <strong>P</strong>ersonalized <strong>A</strong>gentic <strong>S</strong>ocial <strong>S</strong>upport via Tool-Augmented <strong>Com</strong>panionship
</p>

This is the official code repository for the paper *"ComPASS: Towards Personalized Agentic Social Support via Tool-Augmented Companionship"*. 

We hope this work can help shape future companion agents into a "compassionate compass" 🧭, guiding users with personalized support, understanding, and care.

## Table of Contents

- [Preparation of Tool Environment](#preparation-of-tool-environment)
- [ComPASS-Bench Data Synthesis](#compassbench-data-synthesis)
- [ComPASS-Qwen Training](#training)
- [Evaluation](#evaluation)

---

## Preparation of Tool Environment

### Step 1: Environment Installation

Create and activate the conda environment using the following commands:

```bash
conda create -n compass_env python=3.13
conda activate compass_env
pip install -r requirements.txt
```

### Step 2: Model Preparation

1. **CLIP-ViT-Base-Patch32**
   CLIP is used in the tool environment. You need to place the model into the following directory:
   `compass/data_synthesis/code/toolenv/sticker_respond/model`
   *Download link:* [openai/clip-vit-base-patch32 on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
   
2. **All-mpnet-base-v2**
   You need to place the `all-mpnet-base-v2` model into the following directory:
   `compass/data_synthesis/code/toolenv/all-mpnet-base-v2`
   *Download link:* [sentence-transformers/all-mpnet-base-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

### Step 3: Data Preparation

1. **SERdataset**
   Download the dataset from [SuperKogito/SER-datasets on GitHub](https://github.com/SuperKogito/SER-datasets). Once downloaded, place it into the following directory:
   `compass/data_synthesis/code/toolenv/sticker_respond/SERdataset`

### Step 4: Code Preparation

1. Before proceeding, please run the following scripts to form a FAISS index:
   ```bash
   python compass/data_synthesis/code/toolenv/sticker_respond/image2faiss.py
   python compass/data_synthesis/code/toolenv/psyweb_recommender/2faiss.py
   ```

---

## CompassBench Data Synthesis

Run the following script to generate the dataset:
```bash
python compass/data_synthesis/code/persona_gen_main.py
```
*Alternatively, you can use the test data we have already synthesized for testing purposes, located at:* `compass/data_synthesis/data/test_set_correct_form.json`

---

## Evaluation

### Profile-based setting

Run `compass/evaluation/profile-based/code/generate_social_support_respond.py` to generate tool invocation for target model.

Then use `compass/evaluation/profile-based/code/evaluation.py` to evaluate model under persona-based setting.

### History-based setting

Run `compass/evaluation/history-based/code/evaluation_with_his.py` and `compass/evaluation/history-based/code/evaluation_without_his.py` to generate tool invocation with & without interaction history.

Then use `compass/evaluation/history-based/code/evaluation_with_his.py` and `compass/evaluation/history-based/code/evaluation_without_his.py` to evaluate model under history-based setting.

---
## Training

### Step 1: Installation
Please install [ms-swift](https://github.com/modelscope/ms-swift) before starting the training process.

### Step 2: Model Preperation
You have to download the Qwen3-8b model under the direction `compass/train/model` before start training

### Step 3: Start Training
Use the provided shell script to start training:
```bash
bash compass/train/train_qwen3_8b.sh
```
### Step 4: Inference
Use 
```bash
python compass/evaluation/profile-based/code/inference_respond.py
```
for profile-based and 
```bash
python compass/evaluation/history-based/code/inference_toolinvoc_his.py 
python compass/evaluation/history-based/code/inference_toolinvoc_nohist.py
```
for history-based.

---

