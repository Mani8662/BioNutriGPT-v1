"""Generate a Clean, Minimalist Vertical Flowchart for Stage 3 Architecture,
matching the exact design aesthetic, typography, and card style of Stage-2 arch.jpeg.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_simple_stage3_arch():
    # Vertical aspect ratio matching Stage-2 arch.jpeg (~1000 x 2100 px equivalent)
    fig = plt.figure(figsize=(10, 22), dpi=200)
    fig.patch.set_facecolor("#18181b")  # Dark charcoal theme matching Stage 2
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#18181b")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 220)
    ax.axis("off")

    def draw_card(x, y, w, h, title, lines, is_small=False):
        # Outer border / card background
        card = patches.Rectangle(
            (x, y), w, h,
            facecolor="#222226",
            edgecolor="#38383e",
            linewidth=1.2,
            zorder=3
        )
        ax.add_patch(card)

        if is_small:
            # Small pill/badge card like "Pass" or Top/Bottom input
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center", color="#f3f4f6",
                    fontsize=11, weight="bold", zorder=5)
            return

        # Header title
        header_y = y + h - 4.5
        ax.text(x + 3.5, header_y, title,
                ha="left", va="center", color="#9ca3af",
                fontsize=11.5, weight="bold", zorder=5)

        # Thin header separator
        ax.plot([x, x + w], [y + h - 8.0, y + h - 8.0],
                color="#38383e", lw=1.0, zorder=4)

        # Inner dark content box
        inner_margin = 3.0
        inner_w = w - 2 * inner_margin
        inner_h = h - 12.0
        inner_y = y + 2.5
        inner_box = patches.Rectangle(
            (x + inner_margin, inner_y), inner_w, inner_h,
            facecolor="#18181b",
            edgecolor="#2e2e34",
            linewidth=1.0,
            zorder=4
        )
        ax.add_patch(inner_box)

        # Content lines
        curr_y = inner_y + inner_h - 4.0
        for line in lines:
            if line.startswith("Check:") or line.startswith("Solve:") or line.startswith("Input:"):
                ax.text(x + w / 2, curr_y, line,
                        ha="center", va="center", color="#ffffff",
                        fontsize=11, weight="bold", zorder=5)
            elif line.startswith("  -") or line.startswith("  •"):
                ax.text(x + 6.0, curr_y, line,
                        ha="left", va="center", color="#e5e7eb",
                        fontsize=10.5, family="monospace", zorder=5)
            else:
                ax.text(x + w / 2, curr_y, line,
                        ha="center", va="center", color="#d1d5db",
                        fontsize=10.5, zorder=5)
            curr_y -= 4.2

    def draw_pass_badge(x, y, text="Pass"):
        w, h = 18, 7.5
        card = patches.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor="#222226",
            edgecolor="#38383e",
            linewidth=1.0,
            zorder=4
        )
        ax.add_patch(card)
        ax.text(x, y, text,
                ha="center", va="center", color="#9ca3af",
                fontsize=10, zorder=5)

    def draw_down_arrow(x, y_start, y_end):
        ax.annotate(
            "", xy=(x, y_end), xytext=(x, y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#6b7280",
                lw=1.5,
                mutation_scale=12,
            ),
            zorder=2
        )

    # 1. Top Input Card (Stage 2 Output)
    # Y: 200 to 212
    draw_card(
        x=16, y=200, w=68, h=12,
        title="Input from Stage 2: Certified Nutrient Target Vector (N_target)",
        lines=[],
        is_small=True
    )

    draw_down_arrow(50, 200, 185)

    # 2. Block 1: Food Matrix Mapping
    # Y: 147 to 185 (h = 38)
    draw_card(
        x=10, y=147, w=80, h=38,
        title="Step 1: Cultural Food Matrix Mapping (ICMR-NIN IFCT 2017)",
        lines=[
            "Map 39 target nutrients into real Indian food composition matrix:",
            "  - 528 Indian foods analyzed across 151 chemical components",
            "  - Quantized into cooked household serving units (Katoris, Cups, Phulkas)",
            "  - Discrete quantities: q_i in {0, 1, 2, 3} integer servings",
            "  - Fast in-memory local caching (<5 ms loading time, zero API lag)"
        ]
    )

    draw_down_arrow(50, 147, 137)
    draw_pass_badge(50, 137, "Mapped")
    draw_down_arrow(50, 137 - 3.75, 126)

    # 3. Block 2: Discrete MILP Optimization
    # Y: 88 to 126 (h = 38)
    draw_card(
        x=10, y=88, w=80, h=38,
        title="Step 2: Operations Research Integer Solver (HiGHS MILP)",
        lines=[
            "Solve discrete optimization problem to match nutrient envelope:",
            "  - Objective: Minimize caloric & micronutrient deviation with L1 slacks",
            "  - Constraints: Lower <= sum(Food_i * q_i) <= Upper for all 39 nutrients",
            "  - Dynamic penalties: Absorbs conflicting bounds without crashing",
            "  - Hard time-limit budget: time_limit = 0.05s (<7.3 ms average solve)"
        ]
    )

    draw_down_arrow(50, 88, 78)
    draw_pass_badge(50, 78, "Solved")
    draw_down_arrow(50, 78 - 3.75, 67)

    # 4. Block 3: Verification & Culinary Validation
    # Y: 29 to 67 (h = 38)
    draw_card(
        x=10, y=29, w=80, h=38,
        title="Step 3: Verification & Culinary Feasibility Check",
        lines=[
            "Check: Does the discrete food basket satisfy clinical feasibility?",
            "  - Caloric error <= 1.0% relative to Stage 2 energy target",
            "  - Meal diversity bounds: S_min <= Active Dishes <= S_max",
            "  - Solved basket: Exact Katoris of Dal, Sabzi, Rice, Curd & Roti",
            "  - Proven on 17,046 benchmark cases (99.96% empirical feasibility)"
        ]
    )

    draw_down_arrow(50, 29, 19)
    draw_pass_badge(50, 19, "All Verified")
    draw_down_arrow(50, 19 - 3.75, 11)

    # 5. Bottom Output Card
    # Y: 2 to 11
    draw_card(
        x=16, y=2, w=68, h=10,
        title="Certified Indian Food Basket (q*)\nReady for Stage 4 Explanatory LLM Interface",
        lines=[],
        is_small=True
    )

    # Save to disk
    output_jpg = Path("Less Data/Stage-3 arch.jpeg")
    output_png = Path("Less Data/Stage-3 arch.png")
    audit_jpg = Path("Less Data/Audit/Stage-3 arch.jpeg")
    audit_png = Path("Less Data/Audit/Stage-3 arch.png")
    artifact_jpg = Path("C:/Users/MANI KUMAR/.gemini/antigravity/brain/1edf7fa2-4c3a-4ffb-8d91-497686a6f252/Stage-3 arch.jpeg")

    plt.savefig(output_jpg, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.savefig(audit_jpg, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.savefig(audit_png, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.savefig(artifact_jpg, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print("Stage 3 simple architecture diagram generated successfully!")

if __name__ == "__main__":
    generate_simple_stage3_arch()
