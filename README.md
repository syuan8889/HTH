# **HTH: Hierarchical Text-guided Hashing for Open-world Image Retrieval**

Official PyTorch implementation of
**“Hierarchical Text-guided Hashing for Open-world Image Retrieval” (IEEE TCSVT 2026)**

------

## 📌 **1. New Zero-Shot Hashing Benchmark**

🎉We propose a zero-shot hashing benchmark  for image retrieval built upon **ImageNet-CoG**.

### Dataset

1. Download the full ImageNet dataset locally:
   - ImageNet-1K (IN-1K)
   - ImageNet-21K (IN-21K)
2. Levels Splitting (refer to the official repository  [ImageNet-CoG](https://github.com/naver/cog) for more details):
   - **Level 0**: ImageNet-1K (1,000 categories)
   - **Level 1–5**: ImageNet-CoG splits of ImageNet-21K
     Each level contains **1,000 categories**

### **Benchmark Protocol**

- **Training:** Pre-train the model on **Level 0 training set**
- **Evaluation:** Evaluate on **Level 0–5 validation sets**
- **Metric:** Mean Average Precision (**mAP@all**) for each level

------

## ⚙️ **2. Pre-training Pipeline**

### **Stage 1: Hierarchical Text Synthesis**

We use LLaVA to generate hierarchical textual descriptions.

#### **Step 1: Download LLaVA model and Install LLaVA**

llava-hf/llava-v1.6-mistral-7b-hf:
https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

Follow the official install instructions:
https://github.com/haotian-liu/LLaVA#install

#### **Step 2: Generate hierarchical text**

Run the following script to generate the text.

For more details, please refer to https://github.com/Nan-ltl/Hierarchical-text-generation

```bash
python Generate_text_by_llava.py --data_list_path "your path" --output_json "your json file path" --output_error_json "your json file path to record the error images" --data_path "path to dataset" --cache_dir "pretrained model weigths of llava"
```

Example image-text pairs after generating text.

```text
"/n01440764/n01440764_10290.JPEG": "Animal.=>Fish.=>Large fish with blue and yellow pattern.=>The image shows a man holding a large fish with a blue and yellow pattern on its body. He is wearing a white cap, a gray shirt, and blue pants. The man is standing on a dirt path surrounded by green plants and a blue umbrella."
```

------

### **Stage 2: Model Pre-training**

#### **Step 1: Download Fine-tuned CLIP**

We use a **hierarchically fine-tuned CLIP model** from:

**Emergent Visual-Semantic Hierarchies in Image-Text Representations (ECCV 2024)**
Repo: https://github.com/TAU-VAILab/hierarcaps

Download CLIP-B weights from:
https://drive.google.com/drive/folders/1s-f2L0pFzZXs2jCaIyfMa207iiNYkua3

#### Step 2: Environment setup

Please refer to official DINOv2 👉https://github.com/facebookresearch/dinov2

#### Step 3: Run Training Script

We train on **8 × NVIDIA RTX 3090 (24GB)** GPUs. The checkpoints will be saved automatically.

```bash
torchrun --nproc_per_node=8 train.py --output-dir <path to output dir>
```

------

## 🧪 **3. Evaluation**

### **3.1 Dataset Preparation**

- **Level 0**: Use the official ImageNet-1K validation set.

  ```
  ./imagenet-cog/cog_val_level0.txt
  ```

- **Level 1–5**: We extract the validation paths and labels following the official ImageNet-CoG mapping and split files.

  ```text
  ./imagenet-cog/cog_val_level1.txt
  ./imagenet-cog/cog_val_level2.txt
  ./imagenet-cog/cog_val_level3.txt
  ./imagenet-cog/cog_val_level4.txt
  ./imagenet-cog/cog_val_level5.txt
  ```

  👉 [ImageNet-CoG](https://github.com/naver/cog)

Each validation list file contains **50,000 images and corresponding labels**.

Example format for each cog_val_level file:

```text
n01322508/n01322508_2514.JPEG 0
n01322508/n01322508_1956.JPEG 0
n01322508/n01322508_5478.JPEG 0
```

------

### **3.2 Compute Retrieval Performance**

We use the following script to calculate **mAP@all** for each level (0–5).

You can download our pretrained weights here. 👉 https://drive.google.com/drive/folders/12aGY8YYL17ksKq2PnJskTEZmnF_QiUmH?usp=sharing

```bash
python evaluate.py
```

------

## 📚 **Citation**

If you find this work useful, please cite:

```bibtex
@ARTICLE{11339951,
  author={Qiao, Shishi and Yuan, Shuai and Chen, Miaonan and Zheng, Haiyong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Hierarchical Text-guided Hashing for Open-world Image Retrieval}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2026.3651707}}

```

------

## 📜 **License**

This project is released under the MIT License.
