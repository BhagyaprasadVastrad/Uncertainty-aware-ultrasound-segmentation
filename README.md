# Uncertainty-Aware Segmentation in Musculoskeletal Ultrasound

I built a pipeline that segments structures in ultrasound images and also reports how uncertain it is, then tested whether that uncertainty can tell me which segmentations are wrong.

## What this project is

A segmentation model is usually judged by one average accuracy figure. That figure does not say whether the outline on the screen right now can be trusted. My question was whether a model's uncertainty points to the outputs that are wrong, so that uncertain cases could be sent to an expert and confident ones accepted.

My long-term target is synovitis in arthritis ultrasound. I could not find a public synovitis ultrasound dataset with expert masks, so I built and tested the whole pipeline on a public musculoskeletal ultrasound dataset first.

## What I achieved

All results are on 1,643 test images from 273 subjects that the models never saw.

| Result | Value |
|---|---|
| Mean Dice, single U-Net | 0.897 (five seeds: 0.894 to 0.900) |
| Mean Dice, five-model ensemble | 0.902 |
| Failed segmentations (Dice below 0.7) | 43 of 1,643 (2.6%) |
| AUROC, ensemble mutual information | 0.79 (95% interval 0.70 to 0.88) |
| AUROC, ensemble entropy | 0.69 (0.56 to 0.84) |
| AUROC, test-time augmentation mutual information | 0.73 (0.65 to 0.85) |
| AUROC, single-model entropy | 0.55 (0.37 to 0.78) |

- Ensemble disagreement identified failed segmentations much better than one model's own confidence.
- Flagging the 5% most uncertain images (82) caught 17 of the 43 failures. Keeping only the 90% most certain images raised mean Dice from 0.902 to 0.913, and keeping 50% gave 0.922.
- At pixel level the ensemble was better calibrated than a single model (ECE 0.0125 against 0.0194).
- Withholding one muscle from training dropped Dice on that muscle from about 0.89 to between 0.74 and 0.76.
- I found images of other muscles filed in one subject's biceps folder and burned-in text labels on every image I checked. I reported results with and without that subject, and masking the labels changed Dice by no more than run-to-run noise.

## Dataset

Marzola et al., 2021: 8,169 transverse ultrasound images of three muscles (biceps brachii, gastrocnemius medialis, tibialis anterior), healthy and pathological, with expert masks. All images come from one centre and one scanner. The data are open access. Please cite the original paper and follow its terms:

Marzola F, van Alfen N, Doorduin J, Meiburger KM. Deep learning segmentation of transverse musculoskeletal ultrasound images for neuromuscular disease assessment. Computers in Biology and Medicine, 2021. https://doi.org/10.1016/j.compbiomed.2021.104623

## Method

- **Splits:** by subject, never by image (seed 42: 954 train, 136 validation, 273 test subjects). The code stops if any subject appears in two splits.
- **Preprocessing:** grayscale, resized to 256 by 256, scaled to [-1, 1].
- **Model:** U-Net with a ResNet34 encoder (ImageNet weights), binary cross-entropy plus Dice loss, AdamW at 3e-4, batch size 16, up to 30 epochs with early stopping on validation Dice.
- **Uncertainty:** five models trained with different seeds (deep ensemble) and four test-time augmentation passes. Per-pixel entropy and mutual information are averaged into one score per image.
- **Evaluation:** Dice, AUROC for identifying failures, risk-coverage curves, pixel-level ECE and Brier score, and an image-level logistic calibration fitted on validation data only. Intervals come from resampling subjects.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_prepare_data.ipynb` | Builds the subject-level split, checks the data, exports compact arrays |
| `02_train_unet.ipynb` | Trains the U-Net baseline and the shift experiments |
| `03_ensemble_uncertainty.ipynb` | Trains the ensemble and runs the uncertainty analysis |

## How to run

1. Download the dataset and set `ROOT` in notebook 1 to its folder.
2. Run notebook 1. It creates `index.csv` and `data_256.npz`.
3. Run notebooks 2 and 3 on a GPU. I used a Google Colab T4, and one model takes about 20 to 30 minutes.

```
pip install torch torchvision segmentation-models-pytorch pandas numpy pillow scikit-learn scipy matplotlib
```

Trained weights are not included (about 98 MB each).

## Scope

- One scanner, one centre, muscle images with large targets (about a fifth of the image), not synovitis.
- Only 43 failures, so the intervals are wide.
- The 0.7 Dice threshold for a failure is a placeholder.

## Next steps

- Apply the pipeline to synovitis ultrasound data.
- Test uncertainty under centre and device shift.
- Add Monte Carlo dropout.
- Try image-level scores that focus on the predicted region, so small structures are not averaged away.

## Author

Bhagyaprasad Vastrad. 
