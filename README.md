# TEXT-TO-IMAGE-GENRATION-USING-STACK-GANS
TEXT-TO-IMAGE-GENERATION-USING-STACK-GANS
==========================================

PROJECT TITLE:
---------------
Text-to-Image Generation Using StackGANs on the CUB-200-2011 Birds Dataset

DESCRIPTION:
-------------
This project implements a two-stage Generative Adversarial Network (StackGAN) to generate bird images from textual descriptions using the CUB-200-2011 dataset. The architecture is divided into two parts:

1. **Stage 1** – Generates low-resolution (64x64) images from text embeddings.
2. **Stage 2** – Takes Stage 1 outputs and refines them to high-resolution (256x256) images.

Both stages are trained separately using their own models and optimizers.

--------------------------------------------------------------------------------
STAGE 1: (TRAINING FOR 200 EPOCHS)
--------------------------------------------------------------------------------
- File: `stage1_train.ipynb`
- Input: Text embeddings + Noise vector
- Output: 64x64 generated image
- Uses a Conditioning Augmentation Network (CA_NET)
- Trains:
  - `stage1_gen.pth` (Generator)
  - `stage1_dis.pth` (Discriminator)

--------------------------------------------------------------------------------
STAGE 2: (TRAINING AFTER STAGE 1)
--------------------------------------------------------------------------------
- File: `stage_2_colab_ready.ipynb`
- Input: Stage 1 generated image + text embedding
- Output: 256x256 refined image
- Refines the details and enhances realism
- Trains:
  - `stage2_gen.pth` (Generator)
  - `stage2_dis.pth` (Discriminator)
- Requires pretrained Stage 1 generator (`stage1_gen.pth`)

--------------------------------------------------------------------------------
DATASET:
--------------------------------------------------------------------------------
- **Name:** CUB-200-2011 (Caltech-UCSD Birds)
- **Images:** 11,788 images across 200 bird species
- **Captions:** 10 human-written captions per image describing attributes
- **Format:** Images and their associated text descriptions organized per class

--------------------------------------------------------------------------------
REQUIREMENTS:
--------------------------------------------------------------------------------
- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- nltk
- tqdm

NOTE: In Google Colab, upload or mount the dataset and pretrained models manually.

--------------------------------------------------------------------------------
USAGE INSTRUCTION:
--------------------------------------------------------------------------------
1. **Stage 1:**
   - Run `stage1_train.ipynb` to train for 200 epochs.
   - Save the generator and discriminator models as `stage1_gen.pth` and `stage1_dis.pth`.

2. **Stage 2:**
   - Load the pretrained Stage 1 generator.
   - Run `stage_2_colab_ready.ipynb` to train Stage 2 models.
   - Save the outputs as `stage2_gen.pth` and `stage2_dis.pth`.

3. **Image Generation:**
   - Use the final Stage 2 generator to generate 256x256 images from new text inputs.

--------------------------------------------------------------------------------
OUTPUT:
--------------------------------------------------------------------------------
- Generates high-resolution (256x256) bird images from natural language descriptions.

--------------------------------------------------------------------------------
CREDIT:
--------------------------------------------------------------------------------
Based on the paper:  
**"StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks"**  
by Han Zhang et al., CVPR 2017

