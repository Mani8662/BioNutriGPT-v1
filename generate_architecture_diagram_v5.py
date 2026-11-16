"""
BioNutriGPT Complete System Architecture Diagram Generator (v5 - Indian Database & Clinical Pipeline)
Refined with perfect spacing, zero overlaps, and publication-grade dark-mode aesthetic.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_diagram():
    # Setup Figure: 18x28 inches at 200 DPI = 3600 x 5600 pixels
    fig, ax = plt.subplots(figsize=(18, 28), dpi=200)
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#0b0d10')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Color Palette - Sophisticated Modern Dark Theme
    COLOR_BG = '#0b0d10'
    COLOR_STAGE_BG = '#13161c'
    COLOR_BOX_BG = '#1a1e26'
    COLOR_BOX_BORDER = '#29303d'
    COLOR_TITLE = '#ffffff'
    COLOR_TEXT = '#c7cbd4'
    COLOR_MUTED = '#6f7887'
    
    # Accent Highlights
    COLOR_CYAN = '#38bdf8'     # Modalities & Stage 1
    COLOR_PURPLE = '#c084fc'   # Transformers & MoE
    COLOR_AMBER = '#fbbf24'    # Risk heads & Gradients
    COLOR_EMERALD = '#34d399'  # Stage 2 Optimization & Safety
    COLOR_ROSE = '#fb7185'     # Residual Skips
    COLOR_TEAL = '#2dd4bf'     # Stage 3 Indian IFCT & MILP
    COLOR_ORANGE = '#fb923c'   # Stage 4 Culinary LLM
    COLOR_BLUE = '#60a5fa'     # Preferences

    # Helper function to draw rounded boxes
    def draw_card(x, y, w, h, bg_color=COLOR_BOX_BG, border_color=COLOR_BOX_BORDER, radius=0.007, linewidth=1.2, zorder=2):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=bg_color,
                              edgecolor=border_color,
                              linewidth=linewidth,
                              zorder=zorder)
        ax.add_patch(rect)
        return rect

    # Helper function for text inside cards
    def card_text(x, y, title, subtitle=None, title_color=COLOR_TITLE, sub_color=COLOR_TEXT, title_size=11, sub_size=9, ha='center', va='center'):
        if subtitle:
            ax.text(x, y + 0.0055, title, fontsize=title_size, fontweight='bold', color=title_color, ha=ha, va=va, zorder=5, fontfamily='sans-serif')
            ax.text(x, y - 0.0055, subtitle, fontsize=sub_size, color=sub_color, ha=ha, va=va, zorder=5, fontfamily='sans-serif')
        else:
            ax.text(x, y, title, fontsize=title_size, fontweight='bold', color=title_color, ha=ha, va=va, zorder=5, fontfamily='sans-serif')

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, color=COLOR_MUTED, width=1.5, dashed=False, label=None, label_pos=0.5, label_color=COLOR_CYAN, label_size=8.5, label_bg=COLOR_BG):
        style = "Simple,tail_width=0.4,head_width=3.2,head_length=4.0"
        linestyle = '--' if dashed else '-'
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=style,
                                color=color,
                                linestyle=linestyle,
                                linewidth=width,
                                zorder=6)
        ax.add_patch(arrow)
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos
            bbox_props = dict(boxstyle="round,pad=0.25", facecolor=label_bg, edgecolor=color, linewidth=0.8, alpha=0.95)
            ax.text(lx, ly, label, fontsize=label_size, color=label_color, ha='center', va='center', bbox=bbox_props, zorder=7, fontweight='bold')

    # =========================================================================
    # 0. MAIN TITLE & SUBTITLE
    # =========================================================================
    ax.text(0.5, 0.985, "BioNutriGPT: Complete End-to-End Clinical AI Architecture",
            fontsize=20, fontweight='heavy', color='#ffffff', ha='center', va='center', zorder=10)
    ax.text(0.5, 0.971, "Multi-Modal Disease Phenotyping  |  Counterfactual Safety Engine  |  Indian Food (ICMR-NIN IFCT) Solver  |  Clinical LLM",
            fontsize=10.5, color=COLOR_CYAN, ha='center', va='center', zorder=10)

    # =========================================================================
    # 1. INPUT LAYER CONTAINER
    # =========================================================================
    draw_card(0.04, 0.830, 0.92, 0.128, bg_color=COLOR_STAGE_BG, border_color='#232833', radius=0.010, linewidth=1.5, zorder=1)
    ax.text(0.06, 0.945, "INPUT LAYER: 88 Preprocessed ML Features + Clinical Anthropometric Metadata + User Preferences",
            fontsize=11.5, fontweight='bold', color='#94a3b8', ha='left', va='center', zorder=4)

    # 4 Stage 1 Feature Boxes (Top Row of Input Layer)
    feat_y = 0.892
    feat_h = 0.040
    feat_w = 0.165
    draw_card(0.060, feat_y, feat_w, feat_h, border_color='#38bdf844')
    card_text(0.060 + feat_w/2, feat_y + feat_h/2, "Lifestyle, Sleep & Smoking", "[B, 19] Physical, sleep, nicotine", title_size=9.5, sub_size=8, title_color='#e0f2fe')

    draw_card(0.235, feat_y, feat_w, feat_h, border_color='#f43f5e55')
    card_text(0.235 + feat_w/2, feat_y + feat_h/2, "Biomarkers & Lipid Ratios", "[B, 6] HbA1c, Gluc, Lipids, BP", title_size=9.5, sub_size=8, title_color='#ffe4e6')

    draw_card(0.410, feat_y, feat_w, feat_h, border_color='#38bdf844')
    card_text(0.410 + feat_w/2, feat_y + feat_h/2, "24h Diet & Clinical Ratios", "[B, 46] 39 Nutrients + 7 Ratios", title_size=9.5, sub_size=8, title_color='#e0f2fe')

    draw_card(0.585, feat_y, feat_w, feat_h, border_color='#38bdf844')
    card_text(0.585 + feat_w/2, feat_y + feat_h/2, "Demographics & Missing Flags", "[B, 17] Age, Sex, Race, _MISS", title_size=9.5, sub_size=8, title_color='#e0f2fe')

    # User Preferences Box (Right Column of Input Layer)
    pref_x, pref_y, pref_w, pref_h = 0.760, 0.840, 0.185, 0.092
    draw_card(pref_x, pref_y, pref_w, pref_h, bg_color='#151e2b', border_color='#3b82f677')
    ax.text(pref_x + pref_w/2, pref_y + 0.072, "User Preferences & Cuisines", fontsize=9.5, fontweight='bold', color='#60a5fa', ha='center', va='center')
    ax.text(pref_x + pref_w/2, pref_y + 0.052, "Regional Indian Cuisines:", fontsize=8.5, fontweight='bold', color='#cbd5e1', ha='center', va='center')
    ax.text(pref_x + pref_w/2, pref_y + 0.035, "North, South, Gujarati, Bengali", fontsize=8, color='#94a3b8', ha='center', va='center')
    ax.text(pref_x + pref_w/2, pref_y + 0.018, "Dietary: Veg, Jain, Halal, Allergens", fontsize=8, color='#94a3b8', ha='center', va='center')

    # Stage 2 Clinical Metadata Box (Bottom Row of Input Layer)
    meta_x, meta_y, meta_w, meta_h = 0.060, 0.840, 0.690, 0.040
    draw_card(meta_x, meta_y, meta_w, meta_h, bg_color='#14231b', border_color='#10b98166')
    ax.text(meta_x + 0.012, meta_y + meta_h/2 + 0.007, "CLINICAL ANTHROPOMETRIC METADATA:", fontsize=8.5, fontweight='bold', color=COLOR_EMERALD, ha='left', va='center')
    ax.text(meta_x + 0.280, meta_y + meta_h/2 + 0.007, "Weight (kg), Height (cm), Age, Sex, Physical Activity Level", fontsize=8.5, fontweight='bold', color='#f1f5f9', ha='left', va='center')
    ax.text(meta_x + 0.012, meta_y + meta_h/2 - 0.008, "[Anti-Leakage Design] Strictly withheld from Stage 1 to prevent target leakage; ingested by Stage 2 for Mifflin-St Jeor BMR & g/kg protein scaling.", fontsize=7.5, fontstyle='italic', color='#a7f3d0', ha='left', va='center')

    # Clean vertical arrow from 4 Feature Groups into Stage 1
    draw_arrow(0.492, 0.892, 0.492, 0.812, color=COLOR_CYAN, width=2.0)
    ax.text(0.492, 0.822, "88 Features [B, 88]", fontsize=8.5, color='#38bdf8', fontweight='bold', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", facecolor=COLOR_BG, edgecolor=COLOR_CYAN, linewidth=0.8))

    # Long arrow from Metadata bypassing Stage 1 directly down into Stage 2 (Far Left Track)
    draw_arrow(0.080, 0.840, 0.080, 0.582, color=COLOR_EMERALD, width=1.5, dashed=True,
               label="Anthropometrics Metadata (MSJ BMR & Protein/kg)", label_pos=0.55, label_color=COLOR_EMERALD, label_size=8)

    # Long arrow from User Preferences bypassing down into Stage 3 and Stage 4 (Far Right Track)
    draw_arrow(0.852, 0.840, 0.852, 0.352, color='#60a5fa', width=1.5, dashed=True,
               label="Regional Cuisine & Dietary Filters", label_pos=0.48, label_color='#93c5fd', label_size=8)

    # =========================================================================
    # 2. STAGE 1 CONTAINER: MULTIMODAL ENCODER & RISK PREDICTION
    # =========================================================================
    s1_x, s1_y, s1_w, s1_h = 0.14, 0.612, 0.68, 0.198
    draw_card(s1_x, s1_y, s1_w, s1_h, bg_color=COLOR_STAGE_BG, border_color='#38bdf855', radius=0.010, linewidth=1.5, zorder=1)
    ax.text(s1_x + 0.02, s1_y + s1_h - 0.013, "STAGE 1: Multimodal Patient Encoder & Multi-Task Disease Risk Prediction",
            fontsize=11.5, fontweight='bold', color=COLOR_CYAN, ha='left', va='center', zorder=4)

    # Stage 1 - Box 1: FT-Transformer Tokenizer
    b1_y, b1_h = 0.752, 0.035
    draw_card(s1_x + 0.04, b1_y, s1_w - 0.08, b1_h, bg_color='#192230', border_color='#38bdf866')
    card_text(s1_x + s1_w/2, b1_y + b1_h/2,
              "Per-Feature Tokenizer (Feature Tokenizer Transformer)",
              "88 Continuous Features -> 88 Independent Token Embeddings  [B, 88, 128]",
              title_size=10, sub_size=8.5, title_color='#ffffff', sub_color='#93c5fd')
    draw_arrow(s1_x + s1_w/2, b1_y, s1_x + s1_w/2, b1_y - 0.011, color=COLOR_PURPLE, width=1.5)

    # Stage 1 - Box 2: Cross-Modal Transformer Encoder
    b2_y, b2_h = 0.706, 0.035
    draw_card(s1_x + 0.04, b2_y, s1_w - 0.08, b2_h, bg_color='#201c30', border_color='#a855f766')
    card_text(s1_x + s1_w/2, b2_y + b2_h/2,
              "Cross-Modal Transformer Encoder + Modality Attention Pooling",
              "Multi-Head Self-Attention across Modalities -> 4 Fused Modality Vectors  [B, 128]",
              title_size=10, sub_size=8.5, title_color='#ffffff', sub_color='#d8b4fe')
    draw_arrow(s1_x + s1_w/2, b2_y, s1_x + s1_w/2, b2_y - 0.011, color=COLOR_PURPLE, width=1.5)

    # Stage 1 - Box 3: Sparse Adaptive MoE
    b3_y, b3_h = 0.660, 0.035
    draw_card(s1_x + 0.04, b3_y, s1_w - 0.08, b3_h, bg_color='#241a30', border_color='#c084fc66')
    card_text(s1_x + s1_w/2, b3_y + b3_h/2,
              "Sparse Adaptive Mixture-of-Experts (MoE)",
              "4 Tabular Experts with Top-2 Routing & Load-Balancing Loss -> Patient Latent  z_patient  [B, 256]",
              title_size=10, sub_size=8.5, title_color='#ffffff', sub_color='#e9d5ff')
    draw_arrow(s1_x + s1_w/2, b3_y, s1_x + s1_w/2, b3_y - 0.011, color=COLOR_AMBER, width=1.5)

    # Stage 1 - Box 4: Multi-Task Disease Risk Heads & Residual Skips
    b4_y, b4_h = 0.619, 0.030
    draw_card(s1_x + 0.04, b4_y, s1_w - 0.08, b4_h, bg_color='#2c2018', border_color='#f59e0b77')
    card_text(s1_x + s1_w/2, b4_y + b4_h/2,
              "Multi-Task Disease Heads  +  Biological Residual Skip Connections",
              "Calibrated Risk Probabilities:  P(Diabetes) [Cutoff 61.7%]  |  P(Obesity) [44.5%]  |  P(CVD) [60.3%]",
              title_size=10, sub_size=8.5, title_color='#fef08a', sub_color='#fde047')

    # Biological Residual Skip Arrow (arching safely between track and box without overlap)
    skip_arrow = FancyArrowPatch((0.26, 0.840), (s1_x + 0.04, b4_y + b4_h/2),
                                 arrowstyle="Simple,tail_width=0.4,head_width=3.2,head_length=4.0",
                                 connectionstyle="arc3,rad=-0.28",
                                 color=COLOR_ROSE, linestyle='--', linewidth=1.4, zorder=7)
    ax.add_patch(skip_arrow)
    ax.text(0.190, 0.730, "bio_to_diabetes\nbio_to_cvd\n(Direct Skips)", fontsize=7.5, color='#fda4af', fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor=COLOR_BG, edgecolor=COLOR_ROSE, linewidth=0.7, alpha=0.9))

    # Gradient Arrow Connecting Stage 1 to Stage 2
    draw_arrow(0.48, s1_y, 0.48, 0.582, color=COLOR_AMBER, width=2.2,
               label="Counterfactual Risk Gradients: ∇_diet L_risk  [minimize P(Diabetes, Obesity, CVD)]",
               label_pos=0.50, label_color='#fef08a', label_size=9)

    # =========================================================================
    # 3. STAGE 2 CONTAINER: COUNTERFACTUAL NUTRIENT PRESCRIPTION ENGINE
    # =========================================================================
    s2_x, s2_y, s2_w, s2_h = 0.14, 0.382, 0.68, 0.198
    draw_card(s2_x, s2_y, s2_w, s2_h, bg_color=COLOR_STAGE_BG, border_color='#10b98166', radius=0.010, linewidth=1.5, zorder=1)
    ax.text(s2_x + 0.02, s2_y + s2_h - 0.013, "STAGE 2: Counterfactual Nutrient Prescription Engine (SLSQP Optimizer & Filters)",
            fontsize=11.5, fontweight='bold', color=COLOR_EMERALD, ha='left', va='center', zorder=4)

    # Stage 2 - Box 1: Counterfactual Optimization Module
    s2_b1_y, s2_b1_h = 0.525, 0.038
    draw_card(s2_x + 0.04, s2_b1_y, s2_w - 0.08, s2_b1_h, bg_color='#13261e', border_color='#10b98177')
    card_text(s2_x + s2_w/2, s2_b1_y + s2_b1_h/2,
              "Projected Gradient Counterfactual Optimizer (SLSQP / L-BFGS-B)",
              "min L_risk(Diet + ΔN)  +  Smooth Quadratic Barrier Penalty ||ΔN||_W^2  (Bounded by Individual BMR & TDEE)",
              title_size=10, sub_size=8.5, title_color='#ffffff', sub_color='#a7f3d0')
    draw_arrow(s2_x + s2_w/2, s2_b1_y, s2_x + s2_w/2, s2_b1_y - 0.010, color=COLOR_EMERALD, width=1.5)

    # Stage 2 - Box 2: 4 Sequential Safety & Physiological Filters
    s2_b2_y, s2_b2_h = 0.442, 0.065
    draw_card(s2_x + 0.04, s2_b2_y, s2_w - 0.08, s2_b2_h, bg_color='#162c23', border_color='#05966988')
    ax.text(s2_x + s2_w/2, s2_b2_y + s2_b2_h - 0.011, "4 Sequential Verification Filters (Physiological & Medical Envelopes)",
            fontsize=10, fontweight='bold', color='#6ee7b7', ha='center', va='center')
    
    # 4 Filter mini cards inside
    f_w = (s2_w - 0.12) / 4
    f_h = 0.038
    f_y = s2_b2_y + 0.007

    # Filter 1
    draw_card(s2_x + 0.05 + 0*f_w, f_y, f_w - 0.01, f_h, bg_color='#11211b', border_color='#10b98155')
    card_text(s2_x + 0.05 + 0*f_w + (f_w-0.01)/2, f_y + f_h/2, "Filter 1: Atwater", "4*C + 4*P + 9*F = kcal\n(0.00% exact error)", title_size=8.5, sub_size=7, title_color='#ffffff', sub_color='#93c5fd')

    # Filter 2
    draw_card(s2_x + 0.05 + 1*f_w, f_y, f_w - 0.01, f_h, bg_color='#11211b', border_color='#10b98155')
    card_text(s2_x + 0.05 + 1*f_w + (f_w-0.01)/2, f_y + f_h/2, "Filter 2: IOM Safety", "RDA / UL Safe Clamps\n26 Micros + Protein g/kg", title_size=8.5, sub_size=7, title_color='#ffffff', sub_color='#93c5fd')

    # Filter 3
    draw_card(s2_x + 0.05 + 2*f_w, f_y, f_w - 0.01, f_h, bg_color='#11211b', border_color='#10b98155')
    card_text(s2_x + 0.05 + 2*f_w + (f_w-0.01)/2, f_y + f_h/2, "Filter 3: Ratio Coherence", "Sugar/Fiber, Na/K, Sat/Unsat\n(Auto-remediation)", title_size=8.5, sub_size=7, title_color='#ffffff', sub_color='#93c5fd')

    # Filter 4
    draw_card(s2_x + 0.05 + 3*f_w, f_y, f_w - 0.01, f_h, bg_color='#11211b', border_color='#10b98155')
    card_text(s2_x + 0.05 + 3*f_w + (f_w-0.01)/2, f_y + f_h/2, "Filter 4: Closed-Loop", "Stage 1 Rescoring\n(ΔP <= 0 verification)", title_size=8.5, sub_size=7, title_color='#ffffff', sub_color='#93c5fd')

    draw_arrow(s2_x + s2_w/2, s2_b2_y, s2_x + s2_w/2, s2_b2_y - 0.010, color=COLOR_EMERALD, width=1.5)

    # Stage 2 - Box 3: Prescribed Nutrient Targets Output
    s2_b3_y, s2_b3_h = 0.392, 0.030
    draw_card(s2_x + 0.04, s2_b3_y, s2_w - 0.08, s2_b3_h, bg_color='#132d22', border_color='#34d39988')
    card_text(s2_x + s2_w/2, s2_b3_y + s2_b3_h/2,
              "Prescribed 39 Nutrient Targets (N_target) + Caloric Energy + Delta Vector (ΔN)",
              "Personalized Daily Caloric Budget  +  Macronutrients (g)  +  26 Micronutrients (mg/µg)  +  Clinical Ratios",
              title_size=10, sub_size=8.5, title_color='#ecfdf5', sub_color='#6ee7b7')

    # Arrow Connecting Stage 2 to Stage 3
    draw_arrow(0.48, s2_y, 0.48, 0.352, color=COLOR_TEAL, width=2.2,
               label="Target Nutrient Vector N_target [39 Nutrients] + Daily Caloric Target",
               label_pos=0.50, label_color='#5eead4', label_size=9)

    # =========================================================================
    # 4. STAGE 3 CONTAINER: INDIAN CONSTRAINED FOOD SOLVER (MILP) [UPDATED!]
    # =========================================================================
    s3_x, s3_y, s3_w, s3_h = 0.14, 0.178, 0.68, 0.172
    draw_card(s3_x, s3_y, s3_w, s3_h, bg_color=COLOR_STAGE_BG, border_color='#14b8a677', radius=0.010, linewidth=1.5, zorder=1)
    ax.text(s3_x + 0.02, s3_y + s3_h - 0.013, "STAGE 3: Indian Constrained Food Solver (ICMR-NIN IFCT 2017 & MILP)",
            fontsize=11.5, fontweight='bold', color=COLOR_TEAL, ha='left', va='center', zorder=4)

    # Stage 3 - Box 1: ICMR-NIN IFCT 2017 Matrix + RecipeDB
    s3_b1_y, s3_b1_h = 0.280, 0.050
    draw_card(s3_x + 0.04, s3_b1_y, s3_w - 0.08, s3_b1_h, bg_color='#122a28', border_color='#14b8a688')
    ax.text(s3_x + s3_w/2, s3_b1_y + s3_b1_h - 0.011, "ICMR-NIN IFCT 2017 Matrix  +  RecipeDB (IIIT Delhi)",
            fontsize=10.5, fontweight='bold', color='#2dd4bf', ha='center', va='center')
    ax.text(s3_x + s3_w/2, s3_b1_y + s3_b1_h - 0.026,
            "[UPDATED] Replaced US-centric USDA with official National Institute of Nutrition (ICMR-NIN) Indian Tables",
            fontsize=8.5, fontweight='bold', color='#fef08a', ha='center', va='center')
    ax.text(s3_x + s3_w/2, s3_b1_y + s3_b1_h - 0.039,
            "528 Key Indian Foods × 39 Stage-2 Nutrients  |  Cereals, Millets (Ragi, Jowar), Pulses (Dals), Sabzis, Dahi, Oils",
            fontsize=8, color='#99f6e4', ha='center', va='center')
    draw_arrow(s3_x + s3_w/2, s3_b1_y, s3_x + s3_w/2, s3_b1_y - 0.010, color=COLOR_TEAL, width=1.5)

    # Stage 3 - Box 2: Mixed-Integer Linear Program (MILP - PuLP / SciPy)
    s3_b2_y, s3_b2_h = 0.226, 0.036
    draw_card(s3_x + 0.04, s3_b2_y, s3_w - 0.08, s3_b2_h, bg_color='#163130', border_color='#0d948888')
    card_text(s3_x + s3_w/2, s3_b2_y + s3_b2_h/2,
              "Mixed-Integer Linear Program (MILP Optimization via PuLP / SciPy)",
              "min  Σ w_j |Σ q_i F_ij - N_target_j|   subject to:  Discrete Indian Serving Sizes  +  Palatability & Caloric Equality",
              title_size=10, sub_size=8.5, title_color='#ffffff', sub_color='#5eead4')
    draw_arrow(s3_x + s3_w/2, s3_b2_y, s3_x + s3_w/2, s3_b2_y - 0.010, color=COLOR_TEAL, width=1.5)

    # Stage 3 - Box 3: Exact Verified Indian Food Basket Output
    s3_b3_y, s3_b3_h = 0.186, 0.030
    draw_card(s3_x + 0.04, s3_b3_y, s3_w - 0.08, s3_b3_h, bg_color='#193b3a', border_color='#2dd4bf99')
    card_text(s3_x + s3_w/2, s3_b3_y + s3_b3_h/2,
              "Exact Verified Indian Food Basket (JSON)",
              "Grams: Atta/Jowar (75g), Moong Dal (60g), Palak (150g), Low-fat Dahi (150g), Mustard Oil (15g), Spices",
              title_size=10, sub_size=8.5, title_color='#ccfbf1', sub_color='#a7f3d0')

    # Arrow Connecting Stage 3 to Stage 4
    draw_arrow(0.48, s3_y, 0.48, 0.148, color=COLOR_ORANGE, width=2.2,
               label="Mathematically Verified Indian Food Basket (Exact Grams) + Clinical Targets",
               label_pos=0.50, label_color='#fed7aa', label_size=9)

    # =========================================================================
    # 5. STAGE 4 CONTAINER: GROUNDED CLINICAL LLM MEAL PLANNER
    # =========================================================================
    s4_x, s4_y, s4_w, s4_h = 0.14, 0.015, 0.68, 0.130
    draw_card(s4_x, s4_y, s4_w, s4_h, bg_color=COLOR_STAGE_BG, border_color='#f9731666', radius=0.010, linewidth=1.5, zorder=1)
    ax.text(s4_x + 0.02, s4_y + s4_h - 0.013, "STAGE 4: Grounded Clinical LLM Meal Planner (Culturally Authentic Indian Menus)",
            fontsize=11.5, fontweight='bold', color=COLOR_ORANGE, ha='left', va='center', zorder=4)

    # Stage 4 - Box 1: Conditioned Indian Medical & Culinary LLM
    s4_b1_y, s4_b1_h = 0.068, 0.040
    draw_card(s4_x + 0.04, s4_b1_y, s4_w - 0.08, s4_b1_h, bg_color='#291c13', border_color='#f9731688')
    card_text(s4_x + s4_w/2, s4_b1_y + s4_b1_h/2,
              "Conditioned Indian Clinical & Culinary LLM (Gemini 2.5 / Claude / Qwen)",
              "Inputs: Stage 1 Disease Attributions  +  Stage 2 Nutrient Envelope  +  Stage 3 Solved Food Basket  +  Regional Cuisines",
              title_size=10, sub_size=8.5, title_color='#ffffff', sub_color='#fdba74')
    draw_arrow(s4_x + s4_w/2, s4_b1_y, s4_x + s4_w/2, s4_b1_y - 0.010, color=COLOR_ORANGE, width=1.5)

    # Stage 4 - Box 2: Final Output Dossier
    s4_b2_y, s4_b2_h = 0.023, 0.035
    draw_card(s4_x + 0.04, s4_b2_y, s4_w - 0.08, s4_b2_h, bg_color='#332215', border_color='#fb923c99')
    card_text(s4_x + s4_w/2, s4_b2_y + s4_b2_h/2,
              "FINAL OUTPUT: Personalized Indian Clinical Nutrition Dossier",
              "Multi-Disease Risk Report  |  Culturally Authentic Indian Meals (Breakfast, Lunch, Snacks, Dinner)  |  Recipes & Clinical Rationale",
              title_size=10, sub_size=8.5, title_color='#ffedd5', sub_color='#fed7aa')

    # Save diagram
    output_dir = Path(r"c:\Users\MANI KUMAR\CURRENT\GitHub\BioNutriGPT\Models\Less Data")
    output_png = output_dir / "bionutrigpt_architecture_v5.png"
    output_jpg = output_dir / "bionutrigpt_architecture_v5.jpg"
    brain_jpg = Path(r"C:\Users\MANI KUMAR\.gemini\antigravity\brain\1edf7fa2-4c3a-4ffb-8d91-497686a6f252\bionutrigpt_architecture_v5.jpg")

    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_jpg, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(brain_jpg, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Diagram successfully updated and saved to:")
    print(f"  {output_png}")
    print(f"  {output_jpg}")
    print(f"  {brain_jpg}")

if __name__ == "__main__":
    create_diagram()
