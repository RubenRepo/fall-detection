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
| **Random forest (10 trees, depth 5), 5 features** | 0.91 ± 0.07 |	0.89 ± 0.01	| 0.96 ± 0.02 |

The small forest has 450 nodes in total, against about 25,000 for the 200-tree forest, for no decrease in sensitivity and ROC AUC (diference within the error) and a significant but acceptable decrease in specificity, which makes it the candidate for the microcontroller.

The 5 features (`all_mean`, `all_max`, `pre_std`, `all_min`, `pre_max`) were chosen from importance rankings computed on the same working set used for cross-validation, so these scores are slightly optimistic.

**Decision threshold** (small forest, 5 features, out-of-fold predictions on the working set, 356 falls and 934 ADL windows):

| Threshold | Sensitivity | Specificity | Missed falls | False alarms |
| -------- | ------- | ----------- | ------- | ------- |
| 0.3 | 0.95 | 0.81 | 7 | 169 |
| 0.4 | 0.94 | 0.85 | 17 | 135 |
| 0.5 | 0.92 | 0.89 | 33 | 98 |
| 0.6 | 0.88 | 0.92 | 48 | 66 |
| 0.7 | 0.82 | 0.94 | 65 | 50 |

**With a peak trigger**: The intended design is event-triggered: a peak of at least 2.67 g (the 5th percentile of fall peaks) starts the classifier. At a model threshold of 0.4 this catches 328 of 356 falls (92.1%) and produces 100 false alarms (specificity 89.3%). Without the trigger the same model catches 333 falls and produces 139 false alarms, so the trigger removes about 28% of the false alarms at the cost of 5 more missed falls.

**What causes the false alarms**: About 19% of ADL windows reach the trigger, and the model flags about 56% of those (100 of 179). By activity: jogging and fast walking 31, jumping 23, stumbling 15, failing to stand up 13, stairs 9, and lying down, turning while lying, sitting down and standing up 9. With all 12 features the total was similar (109), but jumping accounted for 45 and jogging for 18, so the mix of errors depends on the feature set. A likely reason for these false alarms is that acceleration magnitude carries no information about direction or posture, so a hard landing and a fall can look alike. This is a hypothesis that has not been tested yet.

## Held out evaluation

The 4 held-out participants (110 falls and 398 ADL windows) were evaluated once, with everything fixed beforehand: the 5 features, the small forest, a model threshold of 0.4 and a trigger of 2.67 g. The system caught 105 of 110 falls (sensitivity 0.95) and raised 30 false alarms on 398 ADL windows (specificity 0.92).

| Participant | Falls caught | False alarms (ADL windows) | 
| -------- | ------- | ----------- | 
| 1 | 	27 / 28 | 	4 / 88 | 
| 11 | 	17 / 19 | 6 / 79 | 
| 13 | 	61 / 63 | 9 / 114 | 
| 14 | 	no falls | 11 / 117 | 

These results are in line with cross-validation (0.92 sensitivity and 0.89 specificity with the trigger). They are slightly higher, which is within what a sample of 4 people can produce by chance, and no participant stands out as a failure. Participant 14 has no falls in the held-out set, so only 3 people contribute to the sensitivity. Like everything above, these are per-window results on simulated falls with windows centered on the peak, not an estimate of real-world performance.

## Limitations

- Falls were simulated by 15 healthy volunteers, so real falls in older adults will differ.
- Every training window is centered on the largest peak in its recording. A real device sees arbitrary stretches of movement, so these scores are optimistic for continuous monitoring.
- The false-alarm numbers are per window, not per day. A real rate needs continuous recordings from the final hardware.
- Only a few participants are held out, so any single result will be noisy.
- This is a prototype, meant for a small uni project not a medical device.

## Status

The model and its evaluation are finished. Deployment on an Arduino is future work. The exported header models/fall_model_emlearn.h has not been verified on a board.

## Repo layout

    notebooks/   fall detection notebook
    src/         fall_detection package: windowing, feature extraction, activity names
    data/raw/    dataset files and the dataset's Python script (not included in the repo)
    data/processed/   generated dataframe (not included in the repo)

## Getting the Data

The dataset is not include, fear over copywrite issues. Download it from IEEE DataPort (free account needed) and place the files with the scrips (importantly `FallAllD_2_PYTHON_Structure.py`) in `data/raw/`, else you may need to change diretory on the scrips and notebook. The notebook runs the script, which parses all 6,605 recordings and writes `data/processed/FallAllD.pkl` and `data/processed/FallAllD.h5`. The rest of the notebook loads the `.pkl` file. Only load pickle files you generated yourself, because loading a pickle can execute code.

## Run

Python 3.11 is recommended.

    python -m venv .venv
    source .venv/Scripts/activate      # Git Bash on Windows;
    python -m pip install numpy pandas matplotlib scipy scikit-learn tables jupyter ipykernel emlearn
    jupyter notebook

Then open the notebook in `notebooks/` and run it from the top.

## Citation

If you use the data, cite the dataset paper:

M. Saleh, M. Abbas and R. L. B. Jeannès, "FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical and Deep Learning Applications," IEEE Sensors Journal, vol. 21, pp. 1849-1858, 2021.

## Note on AI Assistance and Authorship

I believe it is important to be transparent about how this project was developed. I used Claude, an AI assistant, for explanations, code suggestions, and drafting documentation. I ran and reviewed the code, and I take responsibility for the analysis and conclusions, which are largely my own.
Although I understand most of the code and can debug and modify it largely independently, I relied on AI assistance more than I would have liked when it came to implementing solutions, particularly deciding what to write and how to translate ideas into code. Consequently, I do not believe this project fully represents my current ability to develop a solution independently from scratch, and I would rather acknowledge that limitation than give a potentially misleading impression of my skills.
I consider this project both an accomplishment and a learning resource. I intend to revisit the implementation, deepen my understanding of the code and the models involved, and improve my ability to implement solutions independently, particularly before undertaking my thesis.
I expect to continue using AI tools in future projects, but I want to become increasingly capable of making implementation decisions independently and using AI as an aid rather than relying on it to bridge gaps in my practical experience. As the author of this project, I take responsibility for the work while acknowledging the role AI assistance played in producing it.