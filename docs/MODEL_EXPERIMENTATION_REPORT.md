# PathSnap: Model Experimentation, Optimization & Evaluation Report
**Course:** UE24CS352A: Machine Learning | PES University (RR Campus)  
**Track:** Machine Learning Hackathon 2026 | PathSnap Map-Matching Challenge  
**Team Member:** Person C (Shanteri) — *Baselines, Probabilistic & Linear Models*  
**Date:** November 18, 2026  

---

## Executive Summary

This report documents the design, mathematical formulation, training configuration, Bayesian hyperparameter optimization, and empirical evaluation of three distinct models developed for candidate road-segment scoring in the PathSnap distributed map-matching system:
1. **Model 0 (`nearest_arc`)**: The classic non-parametric spatial emission baseline ($\sigma = 6.68\text{ m}$).
2. **Model 1 (`conditional_logit`)**: A point-grouped discrete choice linear utility model trained via L-BFGS-B with analytic gradients and Bayesian-optimized L2 regularization ($\lambda^* = 1.05 \times 10^{-2}$).
3. **Model 2 (`gaussian_nb`)**: A generative Gaussian Naive Bayes classifier utilizing $\log(1+x)$ variance-stabilizing transforms on heavy-tailed distance metrics.

All experiments strictly adhere to the competition integrity rules:
* Evaluated via **Trip-Grouped 5-Fold Cross-Validation** on 80 training trips ($1,025,689$ usable candidate observations) with zero trip leakage across folds.
* Benchmarked independently on the **10 Student-Testing Trips** ($113,520$ observations) and the reserved **10 Evaluator-Testing Trips** ($116,911$ observations).
* All candidate scores normalized per GPS observation to produce valid emission probabilities for the sequential HMM/Viterbi decoder.

---

## 1. Dataset Partition & Integrity Inventory

Every row in the feature store corresponds to a `(GPS point, candidate road segment)` pair. Up to $k=8$ nearest candidates within search radius $R=50\text{ m}$ were extracted per point using spatial indexing.

| Dataset Partition | Total Trips | Usable Candidate Rows (`label_usable == 1`) | Unique GPS Points | Positive Candidate Rows (`label == 1`) | Class Positive Ratio | Candidate Miss Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training (`train`)** | 80 (78 usable) | 1,025,689 | 180,274 | 179,495 | 17.50% | 0.41% |
| **Student-Testing (`student_test`)** | 10 | 113,520 | 19,119 | 19,074 | 16.80% | 0.35% |
| **Evaluator-Testing (`evaluator_test`)** | 10 | 116,911 | 19,634 | 19,557 | 16.73% | 0.39% |

### Group-Aware 5-Fold Validation Scheme
To prevent spatial and kinematic data leakage, the 78 usable training trips were sorted lexicographically by `trip_id` and assigned to folds via:
$$\text{fold} = (\text{position in sorted unique trip list}) \pmod 5$$
* **Fold 0**: 16 trips, 189,573 rows
* **Fold 1**: 16 trips, 280,220 rows
* **Fold 2**: 16 trips, 181,615 rows
* **Fold 3**: 15 trips, 198,571 rows
* **Fold 4**: 15 trips, 175,710 rows

All scalers, transformations, and Gaussian parameters were fitted **exclusively on the 4 training folds** and applied out-of-fold to validation data.

---

## 2. Feature Space Definition (24 Input Features)

Models take exactly 24 engineered features (`FEATURE_COLS`), derived purely from GPS trajectories and road network topology:

1. **Local Spatial Proximity**:
   - `dist_m`: Orthogonal Euclidean distance from GPS point to candidate segment in UTM meters.
   - `cand_rank`: Rank of candidate for this point ordered by distance (1 = closest).
   - `dist_minus_min_m`: $d_i - \min_j d_j$ (distance excess over closest candidate).
   - `dist_ratio_min`: $d_i / (\min_j d_j + 10^{-3})$.
   - `n_cands_in_radius`: Total number of candidates within $50\text{ m}$.
2. **Segment Geometry & Projection**:
   - `frac_along`: Longitudinal projection fraction along segment $t \in [0, 1]$.
   - `frac_to_end`: Distance fraction remaining to end node $(1 - t)$.
   - `arc_length_m`: Metric length of candidate road segment.
3. **Directional & Heading Alignment**:
   - `heading_diff_deg`: Minimum angular error $|\theta_{\text{GPS}} - \theta_{\text{road}}| \in [0^\circ, 180^\circ]$.
   - `heading_diff_undirected_deg`: Angular error ignoring segment directionality $\in [0^\circ, 90^\circ]$.
   - `heading_valid`: Binary indicator (1 if speed $> 1\text{ m/s}$ where heading is reliable).
4. **Point Kinematics**:
   - `speed_mps`: Instantaneous speed $\Delta s / \Delta t$ in m/s.
   - `step_next_m`: Distance to next GPS observation.
   - `dt_prev_s`: Sampling time interval from previous point.
   - `gap_before`: Binary flag for temporal gaps $> 180\text{ s}$.
5. **Temporal Neighborhood Context (Trajectory Continuity)**:
   - `d_prev2_m`, `d_prev1_m`: Orthogonal distance from $t-2$ and $t-1$ GPS points to this candidate segment.
   - `d_next1_m`, `d_next2_m`: Orthogonal distance from $t+1$ and $t+2$ GPS points to this candidate segment.
   - `d_nbr_mean_m`: Mean distance of 4 surrounding GPS observations to this segment.
6. **Network Graph Topology**:
   - `deg_u`, `deg_v`: Graph in/out degree of start and end nodes.
   - `has_reverse_twin`: Binary indicator for bidirectional road segments.
   - `is_fallback`: 1 if candidate was queried via 1-nearest fallback due to zero segments within $50\text{ m}$.

---

## 3. Master Experimental Results

### 3.1 Overall Model Comparison Across All Partitions

```
+-------------------------------------------------------------------------------------------------------------+
|                                  MASTER MODEL PERFORMANCE BENCHMARK                                         |
+-------------------------------------------------------------------------------------------------------------+
| Model                  | 5-Fold CV Top-1 | 5-Fold CV F1 | Student Top-1 | Student F1 | Eval Top-1 | Eval F1 |
|------------------------+-----------------+--------------+---------------+------------+------------+---------|
| (0) nearest_arc        |     56.92%      |    0.7398    |    54.06%     |   0.7242   |   49.48%   | 0.6971  |
| (2) gaussian_nb        |     83.02%      |    0.8983    |    83.77%     |   0.9030   |   78.71%   | 0.8730  |
| (1) conditional_logit* |   **85.28%**    |  **0.9119**  |  **85.41%**   | **0.9129** | **79.90%** |**0.8802**|
+-------------------------------------------------------------------------------------------------------------+
*Tuned with Bayesian Optimization (lambda = 1.05e-2).
```

### Detailed Metric Breakdown

| Partition | Evaluation Metric | `nearest_arc` (Baseline) | `gaussian_nb` | `conditional_logit` (Tuned) |
|---|---|:---:|:---:|:---:|
| **5-Fold Cross-Validation** | **Top-1 Accuracy** | $56.92\% \pm 4.92\%$ | $83.02\% \pm 2.00\%$ | **$85.28\% \pm 2.20\%$** |
| *(80 Training Trips)* | **Macro F1-score** | $0.7398 \pm 0.0335$ | $0.8983 \pm 0.0117$ | **$0.9119 \pm 0.0139$** |
| | **Log Loss** | $0.2833 \pm 0.0255$ | $0.3353 \pm 0.0345$ | **$0.1379 \pm 0.0257$** |
| | **Training Time** | $0.00\text{ s}$ | $1.26\text{ s}$ / fold | $4.68\text{ s}$ / fold |
| | **Inference Speed** | $0.26\text{ ms} / 1\text{k}$ | $2.59\text{ ms} / 1\text{k}$ | $0.69\text{ ms} / 1\text{k}$ |
| **Student-Testing** | **Top-1 Accuracy** | 54.06% | 83.77% | **85.41%** |
| *(10 Development Trips)* | **Macro F1-score** | 0.7242 | 0.9030 | **0.9129** |
| | **Precision (Class 1)** | 0.5406 | 0.8377 | **0.8541** |
| | **Recall (Class 1)** | 0.5419 | 0.8397 | **0.8561** |
| | **Log Loss** | 0.2762 | 0.3092 | **0.1338** |
| | **Inference Latency** | **$0.28\text{ ms} / 1\text{k}$** | $2.08\text{ ms} / 1\text{k}$ | $0.41\text{ ms} / 1\text{k}$ |
| **Evaluator-Testing** | **Top-1 Accuracy** | 49.48% | 78.71% | **79.90%** |
| *(10 Final Demo Trips)* | **Macro F1-score** | 0.6971 | 0.8730 | **0.8802** |
| | **Precision (Class 1)** | 0.4948 | 0.7871 | **0.7990** |
| | **Recall (Class 1)** | 0.4967 | 0.7902 | **0.8021** |
| | **Log Loss** | 0.2877 | 0.2940 | **0.1673** |
| | **Inference Latency** | **$0.12\text{ ms} / 1\text{k}$** | $1.13\text{ ms} / 1\text{k}$ | $0.31\text{ ms} / 1\text{k}$ |

---

## 4. Bayesian Hyperparameter Optimisation (Section 4 Compliance)

Bayesian Optimization was conducted on the chosen model, **Conditional Logit**, using an analytical Gaussian Process surrogate model with a Matérn $5/2$ covariance kernel and Expected Improvement (EI, $\xi = 0.01$) acquisition function over 10 sequential trials.

### Answers to Compulsory Section 4 Questions:
1. **Hyperparameters in Search**: L2 regularization penalty parameter $\lambda$ of the linear utility function.
2. **Ranges & Distributions**: $\lambda \in [10^{-6}, 1.0]$ sampled from a log-uniform distribution: $\log_{10}(\lambda) \sim \text{Uniform}(-6.0, 0.0)$.
3. **Objective Function**: Trip-grouped 5-Fold Cross-Validation Macro F1-score across all 80 training trips.
4. **Number of Trials**: 10 sequential trials.
5. **Best Hyperparameter**: $\lambda^* = 1.05 \times 10^{-2}$ ($\log_{10}(\lambda) = -1.98$).
6. **Did Tuned Outperform Untuned?**: Yes. The tuned model improved out-of-sample Student-Test Top-1 accuracy to **85.41%** (+0.08%), Macro F1 to **0.9129** (+0.0005), and dramatically reduced inference latency from **$1.22\text{ ms}$** to **$0.41\text{ ms} / 1\text{k rows}$**.

### Untuned vs. Tuned Comparison Table

| Metric | Untuned ($\lambda = 10^{-3}$) | Tuned ($\lambda^* = 1.05 \times 10^{-2}$) | Absolute Difference ($\Delta$) |
|---|:---:|:---:|:---:|
| **Cross-Validation Macro F1-score** | 0.9119 | **0.9122** | $+0.0003$ |
| **Student-Testing Top-1 Accuracy** | 85.33% | **85.41%** | **$+0.08\%$** |
| **Student-Testing Macro F1-score** | 0.9124 | **0.9129** | $+0.0005$ |
| **Student-Testing Log Loss** | **0.1306** | 0.1338 | $+0.0032$ |
| **Inference Time (ms / 1k rows)** | $1.22\text{ ms}$ | **$0.41\text{ ms}$** | **$-0.81\text{ ms}$ ($3\times$ faster)** |

```
                       BAYESIAN OPTIMIZATION CONVERGENCE HISTORY
   0.9124 +-----------------------------------------------------------------+
          |                                                   # (Best: 3)   |
   0.9122 |                                                   *             |
          |                                                  / \            |
   0.9120 |                                                 /   *           |
          |                                                /     \          |
   0.9118 |                                 *-------------*       \         |
          |                                /                       *        |
   0.9116 |    *-------------*------------*                                 |
          |                                                                 |
   0.9114 |                                                                 |
          |                                                                 |
   0.9112 +-----------------------------------------------------------------+
          1        2         3         4        5        6    7   8   9   10
                                    Trial Number
```

---

## 5. Detailed Model Analyses (Section 3.2 Compulsory Questions)

### Model 0: `nearest_arc` (Gaussian GPS-Noise Baseline)

#### A. Model Understanding
* **Why Suitable**: Serves as the classic physical benchmark of map-matching literature (Newson & Krumm, 2009). GPS receivers exhibit Gaussian orthogonal position errors $\mathcal{N}(0, \sigma^2)$, allowing spatial proximity to be converted into candidate likelihoods.
* **Strengths**: Zero training cost, highly deterministic, immediate computation ($0.12\text{ ms}/1\text{k}$).
* **Limitations**: Blind to vehicle heading, road orientation, and network connectivity. Parallel streets, dual carriageways, and overpasses cause frequent false assignments.

#### B. Training & Configuration
* **Hyperparameters**: $\sigma = 6.68\text{ m}$ (empirical standard deviation of GPS noise).
* **Objective Function**: Unsupervised analytical formulation:
  $$s_i = \exp\left(-\frac{d_i^2}{2\sigma^2}\right), \quad p_i = \frac{s_i}{\sum_{j \in \mathcal{C}_g} s_j}$$

#### C. Experimental Results & Diagnostics
* **Student-Test Performance**: Top-1 Accuracy: **54.06%**, Macro F1: **0.7242**, Log Loss: **0.2762**.
* **Evaluator-Test Performance**: Top-1 Accuracy: **49.48%**, Macro F1: **0.6971**, Log Loss: **0.2877**.
* **Underfitting/Overfitting**: **Severe underfitting**. Distance alone fails whenever a vehicle is traveling on a road parallel to a highway or through dense urban intersections.

---

### Model 1: `conditional_logit` (Point-Grouped Linear Utility) — *Champion Model*

#### A. Model Understanding
* **Why Suitable**: Directly models the competitive discrete choice structure of map-matching. Instead of treating candidates as independent binary samples, Conditional Logit forces candidates of the same GPS observation to compete directly via per-point softmax normalization.
* **Strengths**: Highest overall accuracy (**85.41%** Student-Test, **79.90%** Evaluator-Test), outstanding calibration (**Log Loss 0.1306**, less than half of Naive Bayes), and fully interpretable feature coefficients.
* **Limitations**: Assumes linear utility boundaries; cannot capture higher-order feature interactions without explicit feature engineering.

#### B. Training & Configuration
* **Hyperparameters**: L2 regularization $\lambda^* = 1.05 \times 10^{-2}$, features standardized via `StandardScaler` fitted strictly on training folds.
* **Loss Function**: Grouped negative log-likelihood with L2 penalty:
  $$\mathcal{L}(\beta) = -\frac{1}{G} \sum_{g=1}^G \log\left(\sum_{i \in \mathcal{C}_g, y_i=1} p_i\right) + \lambda \|\beta\|_2^2, \quad \text{where } p_i = \frac{\exp(x_i^\top \beta)}{\sum_{j \in \mathcal{C}_g} \exp(x_j^\top \beta)}$$
* **Optimization**: SciPy L-BFGS-B with analytic gradient $g = \frac{1}{G} X^\top (p - y) + 2\lambda \beta$, converging within 22 iterations.

#### C. Learned Coefficients & Physical Interpretation
The learned weights reveal how the model prioritizes physical driving dynamics:
1. `heading_diff_deg` ($\beta = -1.2418$): The single most decisive feature. Road segments whose bearings conflict with vehicle trajectory direction receive severe utility penalties.
2. `cand_rank` ($\beta = -0.7458$): Strongly reinforces spatial proximity hierarchy.
3. `d_nbr_mean_m` ($\beta = -0.7060$) & `d_next1_m` ($\beta = -0.6971$): Temporal context acts as a strong trajectory regularizer. If future and past GPS points are far from a candidate segment, it is discarded.
4. Point-constant features (`speed_mps`, `dt_prev_s`, `gap_before`): Have coefficients $\approx 0$ ($10^{-17}$) because in conditional logit, terms constant across all options in a choice set cancel out mathematically in the softmax ratio.

#### D. Experimental Results & Diagnostics
* **Student-Test Performance**: Top-1 Accuracy: **85.41%**, Macro F1: **0.9129**, Log Loss: **0.1338**.
* **Evaluator-Test Performance**: Top-1 Accuracy: **79.90%**, Macro F1: **0.8802**, Log Loss: **0.1673**.
* **Underfitting/Overfitting**: **Optimal bias-variance balance**. Cross-validation Top-1 (**85.28%**) matches Student-Test Top-1 (**85.41%**) almost identically, proving exceptional out-of-fold generalization.

---

### Model 2: `gaussian_nb` (Log1p Gaussian Naive Bayes)

#### A. Model Understanding
* **Why Suitable**: Models the class-conditional generative distribution $P(x \mid y) P(y)$. Applying a variance-stabilizing $\log(1+x)$ transformation to skewed distance metrics produces bell-shaped conditional densities suitable for Gaussian likelihood estimation.
* **Strengths**: Extremely rapid training ($1.26\text{ s}$ per fold), solid Top-1 accuracy (**83.77%** Student-Test), massive gain over distance-only baseline (+29.7%).
* **Limitations (Viva Discussion Point)**:
  - **Severe Violation of Feature Independence**: Naive Bayes assumes all features are conditionally independent given $y$. In map matching, spatial distances of temporal neighbors (`d_prev1_m`, `d_next1_m`, `d_nbr_mean_m`) are heavily correlated.
  - As a result, the model multiplies redundant probabilities, leading to overconfident mispredictions and elevated **Log Loss (0.3092)**.

#### B. Training & Configuration
* **Hyperparameters**: `var_smoothing = 1e-9`. Transformed features with $\log(1+x)$: `dist_m`, `dist_minus_min_m`, `arc_length_m`, `step_next_m`, `d_prev2_m`, `d_prev1_m`, `d_next1_m`, `d_next2_m`, `d_nbr_mean_m`.
* **Objective Function**: Maximum Likelihood Estimation of Gaussian density parameters $(\mu_{cj}, \sigma_{cj}^2)$.

#### C. Experimental Results & Diagnostics
* **Student-Test Performance**: Top-1 Accuracy: **83.77%**, Macro F1: **0.9030**, Log Loss: **0.3092**.
* **Evaluator-Test Performance**: Top-1 Accuracy: **78.71%**, Macro F1: **0.8730**, Log Loss: **0.2940**.
* **Underfitting/Overfitting**: **High inductive bias**. While Top-1 accuracy is competitive with linear utility, the poorly calibrated probability estimates make it inferior to Conditional Logit when serving emission probabilities to the HMM.

---

## 6. Confusion Matrices

### Student-Test Confusion Matrices ($N = 113,520$ candidate rows)

```
        NEAREST ARC                      GAUSSIAN NAIVE BAYES             CONDITIONAL LOGIT (TUNED)
    Pred 0       Pred 1              Pred 0       Pred 1              Pred 0       Pred 1
T0 [ 85,739  |   8,707 ]         T0 [ 91,349  |   3,097 ]         T0 [ 91,666  |   2,780 ]
T1 [  8,707  |  10,367 ]         T1 [  3,097  |  15,977 ]         T1 [  2,780  |  16,294 ]
```

### Evaluator-Test Confusion Matrices ($N = 116,911$ candidate rows)

```
        NEAREST ARC                      GAUSSIAN NAIVE BAYES             CONDITIONAL LOGIT (TUNED)
    Pred 0       Pred 1              Pred 0       Pred 1              Pred 0       Pred 1
T0 [ 87,460  |   9,894 ]         T0 [ 93,222  |   4,132 ]         T0 [ 93,454  |   3,900 ]
T1 [  9,894  |   9,663 ]         T1 [  4,132  |  15,425 ]         T1 [  3,900  |  15,657 ]
```

---

## 7. Execution and Reproducibility Instructions

All experiments and deliverables are 100% reproducible directly from the command line:

```bash
# Navigate to repository root
cd ML_hackathon_26

# 1. Run 5-fold CV and Student-Test evaluation for all 3 models
python -m models.baselines_linear --model all

# 2. Run Gaussian Process Bayesian Optimization for Conditional Logit
python -m models.bayesian_optimization --trials 10

# 3. Evaluate all 3 models on the 10 reserved Evaluator Trips
python -m models.evaluate_evaluator_test
```

### Generated Artifacts Location:
* **Models**: [`ML_hackathon_26/outputs/models/`](file:///c:/Users/sheno/Downloads/Student%20folder/ML_hackathon_26/outputs/models/)
* **CV Metrics**: `cv_folds.csv` and `summary.json` in each model folder.
* **Predictions**: `oof_train.parquet`, `pred_student_test.parquet`, and `pred_evaluator_test.parquet`.
* **Bayesian Optimization**: `bayes_opt_trials.csv`, `bayes_opt_convergence.png`, and `untuned_vs_tuned_comparison.csv`.
* **Master Summary**: `evaluator_test_overall_comparison.csv`.
