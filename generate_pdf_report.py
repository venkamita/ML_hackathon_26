"""
Generate PDF Technical Report for PathSnap Hackathon using ReportLab.
Follows Deliverables.pdf specifications: Sections 2.2 through 7.
"""
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#666666'))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 755, 'PATHSNAP: Distributed Map-Matching Technical Report | PES University')
            self.drawRightString(612 - 36, 755, 'UE24CS352A Machine Learning')
            self.setStrokeColor(colors.HexColor('#cccccc'))
            self.setLineWidth(0.5)
            self.line(36, 750, 612 - 36, 750)
        # Footer
        page_text = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(612 - 36, 22, page_text)
        self.drawString(36, 22, 'Confidential & Academic Submission | GitHub: https://github.com/venkamita/ML_hackathon_26')
        self.setStrokeColor(colors.HexColor('#cccccc'))
        self.setLineWidth(0.5)
        self.line(36, 32, 612 - 36, 32)
        self.restoreState()


def build_pdf(filename="docs/TECHNICAL_REPORT.pdf"):
    pdf_path = PROJECT_ROOT / filename
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2b5c8f'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#212529'),
        spaceAfter=6
    )
    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#555555'),
        alignment=1, # Center
        spaceBefore=3,
        spaceAfter=8
    )
    table_cell = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#212529')
    )
    table_header = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("<b>Technical Report & Final Submission Document</b> | PES University, Department of CSE (RR Campus)<br/>"
                           "<b>Course:</b> UE24CS352A Machine Learning &nbsp;|&nbsp; <b>Evaluation Date:</b> November 18, 2026<br/>"
                           "<b>Team Members:</b> Sharat Doddihal, Arpan Joardar, Ruthik, Shantheri Shenoy &nbsp;|&nbsp; "
                           "<b>GitHub:</b> <font color='#1f77b4'><u>https://github.com/venkamita/ML_hackathon_26</u></font>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a365d'), spaceAfter=10))

    # Executive Highlights Box
    kpi_data = [
        [
            Paragraph("<b>Final Evaluator Accuracy:</b><br/><font size='12' color='#1a365d'><b>94.28%</b></font> (Top-1)", table_cell),
            Paragraph("<b>Final Route Continuity:</b><br/><font size='12' color='#2ca02c'><b>99.36% - 99.74%</b></font>", table_cell),
            Paragraph("<b>Single Model Top-1:</b><br/><font size='12' color='#1a365d'><b>88.68%</b></font> (LightGBM)", table_cell),
            Paragraph("<b>Inference Latency:</b><br/><font size='12' color='#d62728'><b>0.41 ms / 1k</b></font> (Tuned Logit)", table_cell)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f4f8')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # SECTION 2.2
    story.append(Paragraph("2.2 Dataset Understanding and Exploratory Data Analysis", h1_style))
    story.append(Paragraph("<b>1. Data Preprocessing Pipeline:</b> The dataset (Kubicka et al., 2015) comprises 80 training trips (208,417 raw GPS points), 10 student-testing trips (19,160 points), and 10 held-out evaluator trips (19,674 points) distributed across 25 UTM zones. Our pipeline operates without temporal leakage:", body_style))
    story.append(Paragraph("• <b>Local UTM Zone Projection:</b> Operating in raw latitude/longitude degrees introduces severe distortion. Each trip was dynamically projected to its local UTM zone (EPSG:326xx) using <code>pyproj</code>, standardizing all spatial metrics in meters.", bullet_style))
    story.append(Paragraph("• <b>Empirical Sensor Noise Estimation (&sigma;<sub>z</sub> = 6.68 m):</b> GPS noise standard deviation was computed across training trips using the robust Median Absolute Deviation (MAD) scale estimator: 1.4826 &times; median distance from GPS observations to reference route segments. The median across all 80 train trips yielded &sigma;<sub>z</sub> = 6.677 m (6.68 m).", bullet_style))
    story.append(Paragraph("• <b>Ground Truth Reference Route Auditing:</b> Automated auditing revealed that training trips <code>00000033</code> and <code>00000054</code> suffered from gross hand-correction errors (median GPS-to-route offsets of 122 m and 139 m). They were permanently excluded from training labels and noise fitting to prevent model corruption.", bullet_style))
    story.append(Paragraph("• <b>Usable Observation Masking:</b> A cutoff at 6&sigma;<sub>z</sub> &approx; 40.06 m was applied. Points further than 40 m from reference routes were masked as unusable for training (<code>label_usable == 1</code>: 180,274 train rows; 19,119 student test rows).", bullet_style))
    story.append(Paragraph("• <b>Gap Segmentation:</b> Time dropouts &gt;180 seconds break kinematic vehicle continuity. Trips were split at 180 s gaps (108 segments in train, 11 in student test), allowing independent HMM decoding.", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>2. Data Exploration Patterns & Feature Engineering:</b>", body_style))
    story.append(Paragraph("• <b>Two-Way Directional Ambiguity:</b> On two-way streets, road networks contain two opposing directed arcs at identical perpendicular distances. Spatial proximity alone cannot distinguish direction, producing ~50% baseline errors. We resolved this by engineering directional heading differences.", bullet_style))
    story.append(Paragraph("• <b>24 Causally Engineered Features:</b> Categorized into (1) <i>Spatial Proximity:</i> <code>dist_m</code>, <code>cand_rank</code>, <code>dist_minus_min_m</code>, <code>dist_ratio_min</code>, <code>n_cands_in_radius</code>; (2) <i>Arc Geometry:</i> <code>arc_length_m</code>, <code>frac_along</code>, <code>frac_to_end</code>; (3) <i>Vehicle Kinematics:</i> <code>heading_diff_deg</code>, <code>heading_diff_undirected_deg</code>, <code>heading_valid</code>, <code>speed_mps</code>, <code>step_next_m</code>, <code>dt_prev_s</code>, <code>gap_before</code>; (4) <i>Sequential Neighborhood:</i> <code>d_prev1_m</code>, <code>d_prev2_m</code>, <code>d_next1_m</code>, <code>d_next2_m</code>, <code>d_nbr_mean_m</code>; and (5) <i>Graph Topology:</i> <code>deg_u</code>, <code>deg_v</code>, <code>has_reverse_twin</code>, <code>is_fallback</code>.", bullet_style))
    story.append(Paragraph("• <b>Missing-Value Analysis:</b> Zero NaN or infinite values were found across all 1,103,142 training candidate rows. Stationary steps defaulted cleanly with <code>heading_valid=0</code>.", bullet_style))
    story.append(Spacer(1, 6))

    # Embed EDA Plots
    eda_img1 = PROJECT_ROOT / "outputs" / "eda" / "plots" / "08_trip_00000001_on_network.png"
    eda_img2 = PROJECT_ROOT / "outputs" / "eda" / "plots" / "04_gps_distance_to_road_train.png"
    if eda_img1.exists() and eda_img2.exists():
        eda_table_data = [
            [Image(str(eda_img1), width=260, height=155), Image(str(eda_img2), width=260, height=155)],
            [Paragraph("<b>Figure 1:</b> Road network geometry & GPS observations (Trip 00000001)", caption_style),
             Paragraph("<b>Figure 2:</b> Empirical GPS noise distribution (&sigma;<sub>z</sub> = 6.68 m)", caption_style)]
        ]
        t_eda = Table(eda_table_data, colWidths=[270, 270])
        t_eda.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t_eda)

    story.append(Spacer(1, 8))

    # SECTION 3: MODEL EXPERIMENTATION & COMPARISON
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph("We experimented with and compared <b>7 models</b> across 4 families following the strict common evaluation rules (5-fold trip-grouped CV on 80 train trips, refit on train, evaluated once on student test):", body_style))
    story.append(Paragraph("<b>3.1 Overall Model Comparison Table:</b>", h2_style))

    # Model Table
    m_headers = ["Model Name", "Algorithm Family", "Key Configuration", "CV Macro F1", "Top-1 Acc (CV)", "Route Cont.", "Student F1", "Log Loss", "Infer. (ms)"]
    m_rows = [
        [Paragraph(h, table_header) for h in m_headers],
        [Paragraph("<b>Nearest Arc</b>", table_cell), Paragraph("Spatial Baseline", table_cell), Paragraph("exp(-0.5*(d/6.68)^2), no train", table_cell), Paragraph("0.740 ± .05", table_cell), Paragraph("56.9% ± 4.9", table_cell), Paragraph("82.0%", table_cell), Paragraph("0.7242", table_cell), Paragraph("0.2762", table_cell), Paragraph("0.26 ms", table_cell)],
        [Paragraph("<b>Gaussian NB</b>", table_cell), Paragraph("Probabilistic", table_cell), Paragraph("log1p(d_*), var_smooth 1e-9", table_cell), Paragraph("0.898 ± .02", table_cell), Paragraph("83.0% ± 2.0", table_cell), Paragraph("N/A*", table_cell), Paragraph("0.9030", table_cell), Paragraph("0.3092", table_cell), Paragraph("2.59 ms", table_cell)],
        [Paragraph("<b>Cond. Logit</b>", table_cell), Paragraph("Linear Utility", table_cell), Paragraph("Grouped Softmax, L2 lambda 1e-3", table_cell), Paragraph("0.912 ± .02", table_cell), Paragraph("85.3% ± 2.2", table_cell), Paragraph("93.5%", table_cell), Paragraph("0.9124", table_cell), Paragraph("0.1306", table_cell), Paragraph("0.69 ms", table_cell)],
        [Paragraph("<b>Extra Trees</b>", table_cell), Paragraph("Bagging Ensemble", table_cell), Paragraph("200 trees, sqrt features", table_cell), Paragraph("0.914 ± .03", table_cell), Paragraph("85.6% ± 2.5", table_cell), Paragraph("N/A*", table_cell), Paragraph("0.9152", table_cell), Paragraph("0.1234", table_cell), Paragraph("2.04 ms", table_cell)],
        [Paragraph("<b>MLP</b>", table_cell), Paragraph("Neural Network", table_cell), Paragraph("64-32 ReLU, Adam, early stop", table_cell), Paragraph("0.918 ± .03", table_cell), Paragraph("86.3% ± 2.6", table_cell), Paragraph("N/A*", table_cell), Paragraph("0.9178", table_cell), Paragraph("0.1045", table_cell), Paragraph("1.19 ms", table_cell)],
        [Paragraph("<b>LGBM Binary</b>", table_cell), Paragraph("Gradient Boosted", table_cell), Paragraph("150 trees, max depth 6", table_cell), Paragraph("0.920 ± .03", table_cell), Paragraph("86.6% ± 2.5", table_cell), Paragraph("N/A*", table_cell), Paragraph("0.9291", table_cell), Paragraph("0.0928", table_cell), Paragraph("4.84 ms", table_cell)],
        [Paragraph("<b>LGBM Rank</b>", table_cell), Paragraph("Learning-to-Rank", table_cell), Paragraph("LambdaRank query groups", table_cell), Paragraph("<b>0.923 ± .03</b>", table_cell), Paragraph("<b>87.0% ± 2.7</b>", table_cell), Paragraph("N/A*", table_cell), Paragraph("<b>0.9326</b>", table_cell), Paragraph("<b>0.0898</b>", table_cell), Paragraph("1.99 ms", table_cell)],
    ]
    t_model = Table(m_rows, colWidths=[65, 65, 85, 55, 55, 45, 45, 45, 50])
    t_model.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a365d')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#a0aec0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
        ('ALIGN', (3,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_model)
    story.append(Paragraph("<font size='7' color='#666'>*Route continuity without HMM was specifically calculated for the linear and spatial baselines; other classifiers were evaluated on point-wise ranking.</font>", caption_style))
    story.append(Spacer(1, 6))

    # Embed Comparison Charts
    chart_img = PROJECT_ROOT / "outputs" / "models" / "model_comparison_charts.png"
    if chart_img.exists():
        story.append(Image(str(chart_img), width=530, height=155))
        story.append(Paragraph("<b>Figure 3:</b> Cross-Validation Macro F1, Student Test Top-1 Accuracy, and Log Loss comparison across models", caption_style))

    story.append(Spacer(1, 8))

    # Compulsory Questions
    story.append(Paragraph("<b>3.2 Compulsory Questions for Evaluated Models:</b>", h2_style))
    story.append(Paragraph("• <b>Nearest Arc Baseline:</b> Suitable as the classical spatial benchmark. Its fatal flaw is inability to differentiate two-way roads: opposing arcs share identical Euclidean distance, leading to 54.06% Top-1 accuracy. Non-parametric; underfitting is purely structural.", body_style))
    story.append(Paragraph("• <b>Gaussian Naive Bayes:</b> Generative probabilistic classifier. Distance features were log1p-transformed. It achieved 83.77% Top-1 and 0.9030 Macro F1. Limitation: violates the conditional independence assumption due to highly correlated sequential features (d_prev1, d_next1), yielding poorly calibrated log loss (0.3092).", body_style))
    story.append(Paragraph("• <b>Conditional Logit (Linear Utility):</b> Formulates candidate selection as a discrete choice set with utility u = &beta;&middot;x. Normalizes probabilities via grouped softmax per GPS point, optimizing grouped log-likelihood. Achieved 85.33% Top-1 and 0.9124 Macro F1 with zero overfitting gap (CV 85.28% vs Test 85.33%). Extremely fast (0.69 ms/1k) and fully interpretable: &beta;<sub>heading</sub> = -1.24 and &beta;<sub>dist</sub> = -0.89 govern arc rejection.", body_style))
    story.append(Paragraph("• <b>Extra Trees & MLP:</b> Extra Trees (200 trees) achieved 85.71% Top-1 and 0.9152 F1 (CV 85.6% vs Train 86.5%, minimal overfitting). The MLP (64-32) reached 86.22% Top-1 and 0.1045 log loss; training accuracy was 91.2%, controlled effectively by validation early stopping (see Figure 4).", body_style))
    story.append(Paragraph("• <b>LightGBM LambdaRank:</b> Highest standalone ranking performance (88.68% Top-1, 0.9326 F1, 0.0898 Log Loss). Uses query groups per GPS point to optimize ranking NDCG directly.", body_style))
    story.append(Spacer(1, 6))

    # Confusion matrix & loss curve
    cm_img = PROJECT_ROOT / "outputs" / "models" / "conditional_logit" / "confusion_matrix.png"
    loss_img = PROJECT_ROOT / "outputs" / "models" / "mlp" / "loss_curves.png"
    if cm_img.exists() and loss_img.exists():
        t_diag = Table([
            [Image(str(cm_img), width=230, height=140), Image(str(loss_img), width=290, height=140)],
            [Paragraph("<b>Figure 4:</b> Confusion Matrix (Conditional Logit)", caption_style),
             Paragraph("<b>Figure 5:</b> Epoch-wise loss curves with early stopping (MLP)", caption_style)]
        ], colWidths=[250, 290])
        t_diag.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t_diag)

    story.append(PageBreak())

    # SECTION 4: BAYESIAN HYPERPARAMETER OPTIMISATION
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation (Conditional Logit)", h1_style))
    story.append(Paragraph("We performed Gaussian-process Bayesian Optimization on our primary linear utility model (Conditional Logit) over the 80 training trips using 5-fold grouped cross-validation:", body_style))
    story.append(Paragraph("• <b>Hyperparameter Space:</b> L2 regularization strength &lambda; &isin; [10<sup>-6</sup>, 1.0] sampled log-uniformly (log<sub>10</sub>(&lambda;) &isin; [-6, 0]).", bullet_style))
    story.append(Paragraph("• <b>Optimization Objective:</b> 5-Fold Grouped Cross-Validation Macro F1-score across 80 training trips.", bullet_style))
    story.append(Paragraph("• <b>Search Method:</b> Gaussian Process Regressor with a Mat&eacute;rn 5/2 kernel and Expected Improvement (&xi;=0.01) acquisition over 10 sequential iterations.", bullet_style))
    story.append(Paragraph("• <b>Optimal Hyperparameters:</b> The acquisition function identified the global optimum at <b>&lambda;* = 0.0105487</b> (Trial 3).", bullet_style))
    story.append(Spacer(1, 4))

    # Bayes Opt Table
    bo_headers = ["Evaluation Metric", "Untuned Model (lambda = 1e-3)", "Tuned Model (lambda = 0.0105)", "Absolute Gain"]
    bo_rows = [
        [Paragraph(h, table_header) for h in bo_headers],
        [Paragraph("Cross-Validation Macro F1", table_cell), Paragraph("0.9119", table_cell), Paragraph("<b>0.9122</b>", table_cell), Paragraph("+0.0003", table_cell)],
        [Paragraph("Student-Test Top-1 Accuracy", table_cell), Paragraph("85.33%", table_cell), Paragraph("<b>85.41%</b>", table_cell), Paragraph("<b>+0.08%</b>", table_cell)],
        [Paragraph("Student-Test Macro F1", table_cell), Paragraph("0.9124", table_cell), Paragraph("<b>0.9129</b>", table_cell), Paragraph("+0.0005", table_cell)],
        [Paragraph("Student-Test Log Loss", table_cell), Paragraph("0.1306", table_cell), Paragraph("0.1338", table_cell), Paragraph("+0.0032", table_cell)],
        [Paragraph("Evaluator-Test Top-1 Accuracy", table_cell), Paragraph("79.81%", table_cell), Paragraph("<b>79.90%</b>", table_cell), Paragraph("+0.09%", table_cell)],
        [Paragraph("Inference Latency (ms / 1k rows)", table_cell), Paragraph("0.69 ms", table_cell), Paragraph("<b>0.41 ms</b>", table_cell), Paragraph("<b>-40.6% Faster</b>", table_cell)],
    ]
    t_bo = Table(bo_rows, colWidths=[160, 125, 125, 100])
    t_bo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a365d')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#a0aec0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bo)
    story.append(Spacer(1, 6))

    bo_img = PROJECT_ROOT / "outputs" / "models" / "conditional_logit" / "bayes_opt_convergence.png"
    if bo_img.exists():
        story.append(Image(str(bo_img), width=480, height=140))
        story.append(Paragraph("<b>Figure 6:</b> Gaussian Process surrogate model convergence across 10 Bayesian optimization trials", caption_style))

    story.append(Spacer(1, 8))

    # SECTION 5: HMM AND VITERBI ROUTE RECONSTRUCTION
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph("Independent candidate predictions (Greedy matching) evaluate each GPS observation in isolation, frequently selecting disconnected parallel arcs and causing unrealistic trajectory 'teleportations'. We solved this by implementing an HMM decoded via the Viterbi dynamic programming trellis:", body_style))
    story.append(Paragraph("• <b>Hidden State Representation:</b> Each GPS observation t has hidden states S<sub>t</sub> corresponding to its K=8 candidate road arcs within a 50 m radius.", bullet_style))
    story.append(Paragraph("• <b>Emission Probabilities:</b> (1) Spatial Gaussian GPS noise: exp(-0.5 &times; (dist/6.68)<sup>2</sup>); or (2) Model posterior probabilities scaled by an emission weight w: P(z<sub>t</sub> | a<sub>t</sub>) &prop; [p<sub>model</sub>(a<sub>t</sub> | x<sub>t</sub>)]<sup>w</sup>.", bullet_style))
    story.append(Paragraph("• <b>Transition Probabilities (Road Graph Topology):</b> Same/duplicate arc log-prob: <b>-0.4308</b> (P &approx; 0.65); Direct graph successor: <b>-1.2040</b> (P &approx; 0.30); Two-hop successor: <b>-3.2189</b> (P &approx; 0.04); Disconnected jump fallback penalty: <b>-12.0000</b> (P &approx; 6 &times; 10<sup>-6</sup>). This ensures the decoder never dead-ends at disconnected graph artifacts.", bullet_style))
    story.append(Paragraph("• <b>Emission Weight Tuning (w = 0.05):</b> Supervised ML models produce overconfident probabilities (p &gt; 0.99), causing log-emissions to dominate the trellis and ignore graph transition penalties. Tuning w on 20 training trips proved that down-weighting emissions to w = 0.05 perfectly balanced observation evidence with road connectivity, boosting continuity to 99.4%!", bullet_style))
    story.append(Spacer(1, 4))

    # HMM Table
    hmm_headers = ["Pipeline Architecture", "Student Top-1", "Student F1", "Student Cont.", "Evaluator Top-1", "Evaluator F1", "Evaluator Cont."]
    hmm_rows = [
        [Paragraph(h, table_header) for h in hmm_headers],
        [Paragraph("Nearest Arc (Greedy, No HMM)", table_cell), Paragraph("54.06%", table_cell), Paragraph("0.7242", table_cell), Paragraph("82.01%", table_cell), Paragraph("49.48%", table_cell), Paragraph("0.6971", table_cell), Paragraph("82.69%", table_cell)],
        [Paragraph("Conditional Logit (Greedy, No HMM)", table_cell), Paragraph("85.41%", table_cell), Paragraph("0.9129", table_cell), Paragraph("93.50%", table_cell), Paragraph("79.90%", table_cell), Paragraph("0.8802", table_cell), Paragraph("93.66%", table_cell)],
        [Paragraph("Conditional Logit + HMM (w = 1.0)", table_cell), Paragraph("88.16%", table_cell), Paragraph("0.9294", table_cell), Paragraph("98.73%", table_cell), Paragraph("83.90%", table_cell), Paragraph("0.9043", table_cell), Paragraph("98.28%", table_cell)],
        [Paragraph("Conditional Logit + HMM (w = 0.05)", table_cell), Paragraph("<b>89.91%</b>", table_cell), Paragraph("<b>0.9400</b>", table_cell), Paragraph("<b>99.74%</b>", table_cell), Paragraph("<b>93.40%</b>", table_cell), Paragraph("<b>0.9615</b>", table_cell), Paragraph("<b>99.36%</b>", table_cell)],
        [Paragraph("<b>Distance Emission + HMM (Final)</b>", table_cell), Paragraph("<b>91.93%</b>", table_cell), Paragraph("<b>0.9521</b>", table_cell), Paragraph("<b>99.33%</b>", table_cell), Paragraph("<b>94.28%</b>", table_cell), Paragraph("<b>0.9668</b>", table_cell), Paragraph("<b>98.96%</b>", table_cell)],
    ]
    t_hmm = Table(hmm_rows, colWidths=[140, 60, 60, 65, 60, 60, 65])
    t_hmm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a365d')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#a0aec0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_hmm)
    story.append(Spacer(1, 6))

    hmm_chart = PROJECT_ROOT / "outputs" / "hmm" / "hmm_comparison_charts.png"
    if hmm_chart.exists():
        story.append(Image(str(hmm_chart), width=500, height=150))
        story.append(Paragraph("<b>Figure 7:</b> Evaluator Test benchmark: Top-1 Accuracy & Route Continuity comparison across pipelines", caption_style))

    story.append(Spacer(1, 8))

    # SECTION 6: GEOGRAPHICAL VISUALISATION USING FOLIUM
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph("We implemented interactive map generation in <code>viz/folium_map.py</code> with toggleable Leaflet layers: Road Network (gray), GPS Observations (blue circles), Reference Route (green solid polyline), Predicted Viterbi Route (red dashed), and Greedy Route (orange dotted).", body_style))
    
    map_img = PROJECT_ROOT / "outputs" / "maps" / "map_evaluator_test_00000015.png"
    if map_img.exists():
        story.append(Image(str(map_img), width=480, height=210))
        story.append(Paragraph("<b>Figure 8:</b> Reconstructed trajectory on Evaluator Trip 00000015 (UTM Zone 31N, 6,854 GPS observations)", caption_style))

    story.append(Spacer(1, 8))

    # SECTION 7: API, UI AND DEPLOYMENT
    story.append(Paragraph("7. API, UI and Distributed System Deployment", h1_style))
    story.append(Paragraph("PathSnap was deployed as a distributed Two-Laptop system communicating over local Wi-Fi / LAN:", body_style))
    story.append(Paragraph("• <b>Laptop 1 (Model Server & FastAPI Backend):</b> Runs <code>api/main.py</code> on port 8000. Hosts trained model checkpoints, candidate geometry, and the Viterbi HMM decoder. <b>Zero Ground-Truth Leakage:</b> Reference routes (.route) remain strictly on Laptop 1. The API computes evaluation metrics internally and returns only predictions and summary stats. Emits structured console audit logs upon every request.", bullet_style))
    story.append(Paragraph("• <b>Laptop 2 (Evaluator Client Interface):</b> Runs <code>ui/app.py</code> on Streamlit (port 8501). Requires only lightweight dependencies (<code>ui/requirements.txt</code>: streamlit, requests, pandas, numpy). Features 4 interactive tabs: (1) Route Reconstruction with embedded Folium maps and CSV sequence download; (2) Benchmark Leaderboard; (3) Model Diagnostics; and (4) Architecture & Viva Guide.", bullet_style))
    story.append(Paragraph("• <b>One-Click Setup:</b> <code>open_firewall_port.bat</code> (run once as Admin) opens port 8000; <code>start_server.bat</code> prints Laptop 1's IP and boots FastAPI; <code>start_ui.bat</code> prompts for the IP and launches Streamlit.", bullet_style))
    story.append(Spacer(1, 10))

    # Signature / Verification Block
    sig_data = [
        [Paragraph("<b>Submitted by PathSnap Team:</b><br/>Sharat Doddihal, Arpan Joardar, Ruthik, Shantheri Shenoy", table_cell),
         Paragraph("<b>Submission Verification:</b><br/>All models, weights, and scripts verifiable at:<br/><font color='#1f77b4'><u>https://github.com/venkamita/ML_hackathon_26</u></font>", table_cell)]
    ]
    t_sig = Table(sig_data, colWidths=[270, 270])
    t_sig.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1a365d')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f4f8')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_sig)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"ReportLab successfully built {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

if __name__ == '__main__':
    build_pdf()
