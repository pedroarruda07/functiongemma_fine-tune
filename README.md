# LoRA Fine-Tuning FunctionGemma 270M for Mobile Actions

<table>
  <tr>
    <td valign="top">

## Overview

This repository contains an adapted version of the original FunctionGemma 270M Mobile Actions fine-tuning
<a href="https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb">notebook</a>
from Google.

<br>

The main goal of this project is to demonstrate how <b>LoRA (Low-Rank Adaptation)</b> and optional <b>QLoRA</b>
can be used to significantly reduce hardware requirements while achieving performance very close to the original
full fine-tuning setup.

<br>

🤗 <b>Try it on Hugging Face:</b> Download the model or grab the <code>.litertlm</code> file and run it directly in Edge Gallery: 

<a href="https://huggingface.co/pmarruda/functiongemma-270m-mobile-actions-litertlm-lora">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Card-yellow" alt="Hugging Face">
</a>

##

The original notebook follows a traditional full fine-tuning recipe that requires an <b>A100 GPU</b>, which means it requires paid Google Colab resources.

This project adapts that notebook to:

<ul>
  <li>Use <b>LoRA adapters</b> instead of full fine-tuning</li>
  <li>Optionally support <b>QLoRA</b> for further memory reduction</li>
  <li>Run on a <b>single NVIDIA T4 GPU</b></li>
  <li>Be <b>free to run</b> on standard Google Colab or Kaggle GPU sessions</li>
</ul>
<br>

  </td>
    <td width="270" align="left" valign="top">
      
  ## Demo in Edge Gallery
      
  <img src="assets/MobileActions_LoRA_Demo.gif" alt="Demo Gif" width="300">
  </td>
  </tr>
</table>

---

## What Changed Compared to the Original

| Aspect | Original Notebook | This Adaptation |
|------|------------------|-----------------|
| Fine-tuning method | Full fine-tuning | LoRA (optional QLoRA) |
| GPU requirement | A100 | Single T4 |
| Cost | Paid (~11$ for Colab units) | Free |
| Training time | ~60 minutes | 3-4 hours |
| Performance | Baseline | Very similar ([Detailed Here](#training-results)) |

The goal was not to redesign the training pipeline, but to **make it more accessible and efficient** without sacrificing results.

---

## Notebooks

This project provides **two equivalent notebooks** in the [/notebooks](./notebooks) directory, adapted for different execution environments:

- <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" alt="Colab" width="20"/> **Google Colab notebook**  
  [colab-finetune_functiongemma_270m_for_mobile_actions_with_LoRA.ipynb](./notebooks/colab-finetune_functiongemma_270m_for_mobile_actions_with_LoRA.ipynb)  
  Includes Colab-specific paths and secret handling.

- <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" alt="Kaggle" width="20"/> **Kaggle notebook**  
  [kaggle-finetune_functiongemma_270m_for_mobile_actions_with_LoRA.ipynb](./notebooks/kaggle-finetune_functiongemma_270m_for_mobile_actions_with_LoRA.ipynb)  
  Ready to run on Kaggle, which provides up to **30 hours of GPU time per week** and avoids reliance on a single long-lived Colab runtime.

Both notebooks walk through the same training and evaluation flow:

1. Environment setup  
2. Dataset preparation  
3. Baseline model loading and testing  
4. LoRA (or QLoRA) fine-tuning  
5. Evaluation and checkpoint saving  
6. Conversion to `.litertlm` for edge deployment  

The original Google notebook can be found in: 
[gemma-cookbook/FunctionGemma](https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb)

---

## Training Results

You can find the raw evaluation outputs used for the results below as JSON files in the **[/results](./results)** directory, including runs for the base model, our LoRA fine-tuned model, and the Google full fine-tuned variant.

**Qualitatively**, the LoRA-fine-tuned model produces structured mobile actions that closely match the behavior of the fully fine-tuned baseline, with no obvious degradation in output format or instruction adherence.

**Quantitavely:**
| Metric | Base Model | Full Fine-Tuning | LoRA (this) |
|------|:------:|:------:|:------:|
| Training Loss (final) | - | 0.008800 | 0.013900 |
| Validation Loss (final) | - | 0.013452 | 0.018618 |
| Correct Tool Accuracy | 0.867846 | 0.954214 | 0.946930 (-0.7%) |
| Exact Match Rate (Tool + Args) | 0.602497 | 0.840791 | 0.790437 (-5%) |

- **Correct Tool Accuracy:** Evaluates if the correct tool was called, even if not with the exact same arguments.
- **Exact Match Rate (Tool + Args):** How many examples got the exact correct function calls. It gives the lower bound of correctness. Some outputs might not match the function arguments exactly but still be acceptable. For example, the show_map function call below:

    - show_map:{'query': 'Maison Marulaz, Besançon, France'}
    - show_map:{'query': 'Maison Marulaz in Besançon, France'}

---

## Training Recipe

- **Base model:** FunctionGemma 270M (instruction-tuned)
- **Fine-tuning method:** LoRA  
  - Optional: QLoRA (configurable in the notebook)
- **Dataset:** Mobile Actions dataset
- **Framework:** Hugging Face TRL & PEFT
- **Optimizer:** paged_adamw_32bit (optimized for LoRA and QLoRA)
- **Learning rate:** 1e-5
- **Batch size:** 4
- **Gradient accumulation steps:** 8
- **Number of epochs:** 3
- **LoRA configuration:**
  - Rank (r): 16
  - Alpha: 32
  - Target modules: All Linear ('q_proj', 'down_proj', 'gate_proj', 'v_proj', 'up_proj', 'o_proj', 'k_proj')
  - lora_dropout: 0.1
  - bias: "none"
  - task_type: "CAUSAL_LM"

### Hardware

- **GPU:** Single NVIDIA T4
- **Training environment:** Google Colab or Kaggle
- **Training time:** 1h 45 minutes per epoch

For reference, the original notebook required an A100 GPU and paid Colab resources, whereas this adaptation runs on free Colab or Kaggle GPU sessions.

---

## Edge Deployment

The notebook includes a step to convert the trained checkpoint into a **`.litertlm`** file.

You can also download it directly from [Hugging Face](https://huggingface.co/pmarruda/functiongemma-270m-mobile-actions-litertlm-lora)

This file can be:
1. Uploaded directly to **Edge Gallery** in the 'Mobile Actions' section
2. Run on supported edge devices without additional conversion

---

## Why This Matters

This project shows that:
- Efficient fine-tuning methods like LoRA can dramatically lower barriers to entry
- Strong results do not require expensive hardware
- Edge and mobile-focused models can be trained and deployed with modest resources

---

## Attribution and License

This project is **based on and adapted from the original FunctionGemma Mobile Actions fine-tuning notebook** published by Google.

- **Original work:** © 2025 Google LLC  
- **License:** Apache License, Version 2.0  
- **Original notebook:** [gemma-cookbook/FunctionGemma](https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb)

Modifications in this repository include:
- Introduction of LoRA and optional QLoRA fine-tuning
- Reduced hardware requirements (A100 → T4)
- Minor training and configuration adjustments
- Added results json files.

This repository is distributed under the **Apache License 2.0**, in compliance with the original license.

---

## Disclaimer

This is an independent adaptation and is not an official Google product or release.
