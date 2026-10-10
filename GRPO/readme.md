# 🚀🚀🚀 simple_GRPO 🚀🚀🚀
A very simple GRPO implement for reproducing r1-like LLM thinking.
This is a simple open source implementation that utilizes the core loss calculation formula referenced from Hugging Face's trl. 
We make the simplest codebase to support: 
- Save the GPU memory to make a feasible and efficient training. 
- Quickly understand RL processes such as GRPO from a teaching perspective. 
- Quickly try a lot of things, such as improved multi-answer generation, regrouping, penalty on KL, and parameter tuning.
- "Aha moment" is observed during the early stages of model training.

## Usage

### JUST two py files, ref_server.py and grpo_ref_split.py are enough!
Run the following command:
``` bash
CUDA_VISIBLE_DEVICES=7 python ref_server.py
```
This just uses one GPU to collect and run the reference model.
We use http to transport data and logits.
They have so little data that they won't be any bottlenecks, http is the easiest to understand and has the fewest dependencies, and can be easily supported on multiple machines (we are using dozens of outdated 4090s to generate QA pairs for them!).

Then, open another bash:
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 deepspeed grpo_ref_split.py
```
Use all other GPUs for training!

All parameters are in the code. We need to try more possibilities than a fking long argparse.

## ✨ Experimental Results

1. Runtime Environment
- Hardware Setup: 2×A800 (80GB) GPUs
- Configuration:
  - Training: 1 GPU with Zero-Stage 2 optimization
  - Inference: Dedicated 1 GPU (3090/4090 compatible)

2. Training Performance
   
| Model        | Steps | Time       |
|--------------|-------|------------|
| Qwen2.5-3B   | 60    | 12m 34s    |
| Qwen2.5-7B   | 60    | 16m 40s    |

2.1 Qwen2.5-3B

<img src="https://github.com/lsdefine/simple_GRPO/blob/main/images/Qwen2dot5-3B-res.jpg" alt="Qwen2.5-3B Accuracy" width="70%" style="display: inline-block; margin-right: 5px;">  

- Accuracy:
  - Stabilizes above 60% after 5 optimization steps
  - Peaks at ~70%
- Format Compliance:
  - Reaches ≈100% after 30 steps

2.2 Qwen2.5-7B

<img src="https://github.com/lsdefine/simple_GRPO/blob/main/images/Qwen2dot5-7B-res.jpg" alt="Qwen2.5-3B Accuracy" width="70%" style="display: inline-block; margin-right: 5px;">  

- Accuracy:
  - Maintains >90% throughout training
- Format Compliance:
  - Achieves 100% within 30 steps
- Convergence: Both metrics show rapid convergence

## Aha Moment
1. Qwen2.5-3B (Step 20)
> `<think>`
> ...
> To find out how many more eggs Camden painted than Arnold,  we subtract the number of eggs painted by Arnold from the number of eggs painted by Camden. So, 21 - 28 = -7.  **However, this result doesn't make sense in the context of the problem,
> as Camden cannot have painted fewer eggs than Arnold.
> Let's reconsider the initial solution steps:
> we should verify if the initial assumption about the relationship between Camden and Sarah's eggs is correct.**
> ...
> `</think>`
> `<answer>`-7`</answer>`

2. Qwen2.5-7B (Step 20)

> `<think>`
> ...
> Therefore, Joanne gathered 350 apples from the average trees.
> **However, this doesn't seem right because the total should be 500 and we've already accounted for 150,
> leaving room only for 350 from the average trees, which contradicts the total. Let's reassess.**
> ...
> `</think>`
> `<answer>`350`</answer>`