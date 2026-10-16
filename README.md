## Fall detection from a waist accelerometer (FallAllD)

A small machine-learning project that classifies falls vs. activities of daily living (ADLs) from waist-worn accelerometer data, designed to run on an Arduino-class microcontroller.

**Status**:Work in progress

## Goal

Train a lightweight model on the FallAllD dataset for use in an waist accelerometer Arduino setup.

## Pipeline

1. **Data**: FallAllD waist accelerometer recordings (1,798 trials: 466 falls, 1,332 ADLs, 15 participants). Raw 16-bit counts are converted to g. Activity converted to a binary (0 = Safe / Activity of Daily Living, 1 = Fall) .
2. **Windowing**: each 20 s recording(238Hz) is downsampled to 50Hz, converted into magnitude and cut into a 3 second window (1 second before the fall and 2 seconds after)
3. **Features**: summary statistics (max, min, std, mean) of the whole window, pre-impact (first second) and post-impact (0.5 seconds after impact), 12 features in total (3 windows * 4 statistics)
4. **Models compared**: logistic regression, random forest (200 trees), a small random forest (10 trees, depth 5) and a single decision tree (depth 4).
5. **Evaluation**: 5-fold cross-validation split by participant (StratifiedGroupKFold), so no person appears in both training and test data. A separate 25% of participants is held out and has not been used yet.

## Results

| Model | Sensitivity | Specificity | ROC AUC |
| -------- | ------- | ----------- | ------- |
| Logistic regression | 0.88 ± 0.11 | 	0.90 ± 0.05 | 0.95 ± 0.03 |
| Random forest (200 trees) | 0.90 ± 0.06 | 0.95 ± 0.02 | 0.98 ± 0.01 |
| Random forest (10 trees, depth 5) | 0.90 ± 0.06 | 0.90 ± 0.02 | 0.97 ± 0.01 |
| Decision tree (depth 4) | 0.90 ± 0.06 | 0.87 ± 0.02 | 0.91 ± 0.02 |

The small forest has 450 nodes in total, against about 25,000 for the 200-tree forest, for no decrease in sensitivity and ROC AUC (diference within the error) and a significant but acceptable decrease in 	specificity, which makes it the candidate for the microcontroller.

**Decision threshold** (small forest, out-of-fold predictions on the working set, 356 falls and 934 ADL windows):

| Threshold | Sensitivity | Specificity | Missed falls | False alarms |
| -------- | ------- | ----------- | ------- | ------- |
| 0.3 | 0.98 | 0.82 | 7 | 169 |
| 0.4 | 0.95 | 0.86 | 17 | 135 |
| 0.5 | 0.91 | 0.90 | 33 | 98 |
| 0.6 | 0.87 | 0.93 | 48 | 66 |
| 0.7 | 0.82 | 0.95 | 65 | 50 |

**With a peak trigger**: The intended design is event-triggered: a peak of at least 2.67 g (the 5th percentile of fall peaks) starts the classifier. At a model threshold of 0.4 this catches 330 of 356 falls (92.7%) and produces 109 false alarms (specificity 88.3%). Without the trigger the same model catches 339 falls and produces 135 false alarms. So the trigger removes about a fifth of the false alarms at the cost of 9 more missed falls.

**What causes the false alarms About 19% of ADL windows reach the trigger, and the model flags about 61% of those (109 of 179). Jumping (45), stumbling (15) and failing to stand up (12) account for 72 of the 109. A likely reason is that acceleration magnitude carries no information about direction or posture, so a hard landing and a fall can look alike. This is a hypothesis that has not been tested yet.

## Limitations

- Falls were simulated by 15 healthy volunteers, so real falls in older adults will differ.
- Every training window is centered on the largest peak in its recording. A real device sees arbitrary stretches of movement, so these scores are optimistic for continuous monitoring.
- The false-alarm numbers are per window, not per day. A real rate needs continuous recordings from the final hardware.
- Only a few participants are held out, so any single result will be noisy.
- This is a prototype, meant for a small uni project not a medical device.

## Setup

- Evaluate once on the held-out participants, with per-subject results.
- Measure the false-alarm rate on continuous recordings.
- Export the small model to C and run it on the Arduino.
- *Maybe for future refining*: Add posture-change features (direction of gravity before vs. after impact) using the three axes, which should help separate jumps from falls. This makes the model depend on how the board is mounted.

## Repo layout

    notebooks/   fall detection notebook
    src/         fall_detection package: windowing, feature extraction, activity names
    data/raw/    dataset files and the dataset's Python script (not included in the repo)
    data/processed/   generated dataframe (not included in the repo)

## Getting the Data

The dataset is not include, fear over copywrite issues. Download it from IEEE DataPort (free account needed) and place the files with the scrips (importantly `FallAllD_2_PYTHON_Structure.py`) in `data/raw/`, else you may need to change diretory on the scrips and notebook. The notebook runs the script, which parses all 6,605 recordings and writes `data/processed/FallAllD.pkl`. The rest of the notebook loads that file.

## Run

Python 3.11 is recommended.

    python -m venv .venv
    source .venv/Scripts/activate      # Git Bash on Windows;
    python -m pip install numpy pandas matplotlib scipy scikit-learn tables jupyter ipykernel
    jupyter notebook

Then open the notebook in `notebooks/` and run it from the top.

## Citation

If you use the data, cite the dataset paper:

M. Saleh, M. Abbas and R. L. B. Jeannès, "FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical and Deep Learning Applications," IEEE Sensors Journal, vol. 21, pp. 1849-1858, 2021.

## Author

Ruben Barbosa
