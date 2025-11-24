# CTRG
Implemention of paper 《Structure Observation Driven Image-Text Contrastive Learning for Computed Tomography Report Generation》


# Data Preparation

1. Download the dataset from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).
2. Download the CT-CLIP model from [https://github.com/ibrahimethemhamamci/CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP).
3. Download the pretrained text encoder: [**CXR-BERT-general**](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-general).
4. Download the LLM text decoder: [**LLaMA-2-7B**](https://huggingface.co/meta-llama/Llama-2-7b).

# Environment Setup

1. Install dependencies:  
   ```bash
   pip install -r requirements_final.txt

2. Install CT-CLIP package following its official instructions.
3. install ctvit:
   ```bash
   cd ctvit
   pip install -e .

# training

1. Pretrain: Update the model and data paths in /CTRG/config.py, run
   ```bash
   ./run_scripts/pretrain_3D.sh
   
3. To reduce memory consumption during finetuning, extract visual features after pretraining:
   Set the pretrained model path in pretrained_visual_feature_extract.py, run
  ```bash
  ./run_scripts/visual_feature_extract.sh

5. Finetuning: Update /CTRG/config.py with the correct model and data paths (including the pretraining checkpoint from step 1).
  ```bash
  ./run_scripts/finetune_rg.sh

# Inference
  Run the inference script:
```bash
./run_scripts/finetune_rg_test.sh

A .txt file containing the generated reports will be saved in the checkpoint folder. You can then compute your desired evaluation metrics (e.g., BLEU, ROUGE, etc.) on this output.


