# GANs
This notebook implements a CycleGAN model to perform artistic style transfer from real-world photographs to Claude Monet-style paintings.

The dataset includes:
- `monet_jpg/`: 1,000 Monet paintings (256×256 RGB)
- `photo_jpg/`: 6,000 real-world photos (256×256 RGB)

There are no labels. The model is evaluated using **MiFID**, which penalizes both low realism and overfitting to training data.

---

## Exploratory Data Analysis (EDA)

- Sampled and displayed 5 Monet paintings and 5 photos
- Verified all images are 256×256 RGB with pixel values in [0, 255]
- Observed stylistic differences in texture and color, guiding model choice

---

## Model Architecture and Training Strategy

**CycleGAN architecture:**
- Two generators: `G` (photo → Monet), `F` (Monet → photo)
- Two discriminators: `D_X` (Monet), `D_Y` (photo)
- Losses: adversarial, cycle consistency, identity

**Training parameters:**
- Dataset size: 300 Monet/photo images
- Epochs: 10
- Batch size: 1
- Generator uses 1 residual block for speed
- Optimizer: Adam with learning rate 2e-4 and β₁ = 0.5

Training completed in ~30 minutes on Kaggle GPU.

---

## Results and Submission

- Visual inspection shows clear transformation to Monet style
- Exported 7,000 Monet-style images using trained generator
- Saved to `images.zip` as required for submission
- Received a valid MiFID score on the public leaderboard

---

## Reflections

- CycleGAN worked well even with limited data
- Reduced model complexity helped with fast training
- Larger dataset and more residual blocks would improve fidelity
- Future work could include full training on 6,000 images and training for 20+ epochs
