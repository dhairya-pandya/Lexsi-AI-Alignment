"""
The linear algebra of the low-rank tamper repair, in a 3Blue1Brown-style
manim animation. No LaTeX (all text is Pango), so it renders without a TeX
install.

Render:
    manim -qh assets/anim/repair_scene.py LowRankRepair
"""

import numpy as np
from manim import *

# Palette (Apple system colors on a near-black canvas)
BG      = "#0E0E12"
INK     = "#F2F2F7"
SUB     = "#9A9AA2"
GREEN   = "#34C759"
RED     = "#FF3B30"
BLUE    = "#0A84FF"
AMBER   = "#FF9F0A"
HAIR    = "#3A3A40"

config.background_color = BG
FONT = "Helvetica Neue"  # Pango falls back gracefully if absent


def txt(s, size=36, color=INK, weight=NORMAL, slant=NORMAL):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, slant=slant)


def heat(values, vmin, vmax, base_color, cell=0.52, gap=0.06):
    """A grid heatmap of a 2D array -> VGroup of squares, opacity by magnitude."""
    n, m = values.shape
    g = VGroup()
    for i in range(n):
        for j in range(m):
            t = (abs(values[i, j]) - vmin) / (vmax - vmin + 1e-9)
            t = float(np.clip(t, 0.0, 1.0))
            sq = Square(side_length=cell, stroke_width=0)
            sq.set_fill(base_color, opacity=0.08 + 0.92 * t)
            sq.move_to(np.array([j * (cell + gap), -i * (cell + gap), 0.0]))
            g.add(sq)
    g.move_to(ORIGIN)
    return g


class LowRankRepair(Scene):
    def construct(self):
        self.title_card()
        self.act_update()
        self.act_svd()
        self.act_geometry()
        self.act_projection()
        self.act_result()
        self.end_card()

    # -- 0. title -------------------------------------------------------------
    def title_card(self):
        ey = txt("CROSS-LINGUAL TAMPER RESISTANCE", 22, BLUE, weight=BOLD)
        t1 = txt("How a low-rank repair", 56, INK, weight=BOLD)
        t2 = txt("made the model safe again", 56, INK, weight=BOLD)
        sub = txt("the linear algebra of the defense", 30, SUB)
        ey.to_edge(UP, buff=1.3)
        grp = VGroup(t1, t2).arrange(DOWN, buff=0.15).move_to(ORIGIN)
        sub.next_to(grp, DOWN, buff=0.5)
        self.play(FadeIn(ey, shift=DOWN * 0.3), run_time=0.8)
        self.play(Write(grp), run_time=1.6)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.8)
        self.wait(1.2)
        self.play(FadeOut(VGroup(ey, grp, sub)), run_time=0.7)

    # -- 1. the weight update -------------------------------------------------
    def act_update(self):
        head = txt("A fine-tune attack changes the weights", 34, INK, weight=BOLD).to_edge(UP, buff=0.7)
        self.play(FadeIn(head, shift=DOWN * 0.2), run_time=0.7)

        rng = np.random.default_rng(7)
        n = 7
        Wb = rng.normal(0, 1, (n, n))
        hb = heat(Wb, 0, 2.4, BLUE).scale(0.9)
        lb = txt("W_base", 28, BLUE).next_to(hb, DOWN, buff=0.3)
        g_base = VGroup(hb, lb).to_edge(LEFT, buff=1.0)

        # a nearly rank-1 update: one dominant outer product + faint noise
        u = rng.normal(0, 1, n); v = rng.normal(0, 1, n)
        dW = 2.6 * np.outer(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        dW += rng.normal(0, 0.06, (n, n))
        Wp = Wb + dW
        hp = heat(Wp, 0, 2.4, GREEN).scale(0.9)
        lp = txt("W_post", 28, GREEN).next_to(hp, DOWN, buff=0.3)
        g_post = VGroup(hp, lp).to_edge(RIGHT, buff=1.0)

        arrow = Arrow(g_base.get_right(), g_post.get_left(), buff=0.3, color=AMBER, stroke_width=5)
        alab = txt("attack", 24, AMBER).next_to(arrow, UP, buff=0.15)

        self.play(FadeIn(g_base, shift=RIGHT * 0.2), run_time=0.8)
        self.play(GrowArrow(arrow), FadeIn(alab), run_time=0.7)
        self.play(FadeIn(g_post, shift=LEFT * 0.2), run_time=0.8)
        self.wait(0.6)

        # collapse to dW = W_post - W_base
        eq = txt("dW  =  W_post  −  W_base", 34, INK, weight=BOLD)
        hd = heat(dW, 0, 2.4, RED).scale(0.9)
        ld = txt("dW", 28, RED).next_to(hd, DOWN, buff=0.3)
        g_d = VGroup(hd, ld)
        eq.to_edge(UP, buff=0.8)
        g_d.move_to(ORIGIN).shift(DOWN * 0.3)

        self.play(FadeOut(VGroup(g_base, g_post, arrow, alab, head)), run_time=0.7)
        self.play(FadeIn(eq, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(g_d, scale=0.9), run_time=0.8)
        note = txt("The whole update is one smooth pattern: it is low rank.", 26, SUB)
        note.next_to(g_d, DOWN, buff=0.45)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.8)
        self.wait(1.4)
        self.play(FadeOut(VGroup(eq, g_d, note)), run_time=0.7)

    # -- 2. SVD spectrum ------------------------------------------------------
    def act_svd(self):
        head = txt("Decompose it:  dW = Σ σᵢ · uᵢ vᵢᵀ", 36, INK, weight=BOLD).to_edge(UP, buff=0.8)
        self.play(FadeIn(head, shift=DOWN * 0.2), run_time=0.8)

        energy = np.array([0.70, 0.21, 0.05, 0.02, 0.012, 0.006, 0.002])
        energy = energy / energy.max()
        bars = VGroup()
        labels = VGroup()
        bw, maxh, gap = 0.9, 3.4, 0.35
        x0 = -((len(energy) - 1) * (bw + gap)) / 2
        for i, e in enumerate(energy):
            col = RED if i == 0 else (AMBER if i == 1 else HAIR)
            b = Rectangle(width=bw, height=maxh * e, stroke_width=0).set_fill(col, opacity=1.0)
            b.move_to(np.array([x0 + i * (bw + gap), -1.5 + maxh * e / 2, 0]))
            bars.add(b)
            labels.add(txt(f"σ{i+1}" if i < 3 else "", 22, SUB).next_to(b, DOWN, buff=0.2))
        baseline = Line(bars.get_left() + LEFT * 0.3, bars.get_right() + RIGHT * 0.3, color=HAIR).move_to(
            np.array([0, -1.5, 0]))

        self.play(Create(baseline), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.12), run_time=1.4)
        self.play(FadeIn(labels), run_time=0.5)

        brace = Brace(VGroup(bars[0], bars[1]), DOWN, color=INK)
        bl = txt("top 2 directions hold ≈ 50% of the energy", 26, INK).next_to(brace, DOWN, buff=0.2)
        self.play(GrowFromCenter(brace), FadeIn(bl), run_time=0.9)
        tag = txt("nearly rank-1", 30, RED, weight=BOLD).next_to(bars[0], UP, buff=0.3)
        self.play(FadeIn(tag, shift=DOWN * 0.2), run_time=0.7)
        self.wait(1.4)
        self.play(FadeOut(VGroup(head, bars, labels, baseline, brace, bl, tag)), run_time=0.7)

    # -- 3. geometry: concentrated harm vs diffuse benign ---------------------
    def act_geometry(self):
        head = txt("The harmful write points one way. The benign gain is spread out.",
                   28, INK, weight=BOLD).to_edge(UP, buff=0.7)
        self.play(FadeIn(head, shift=DOWN * 0.2), run_time=0.8)

        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": HAIR, "stroke_width": 1, "stroke_opacity": 0.5},
            axis_config={"stroke_color": SUB, "stroke_width": 2},
        ).scale(0.95).shift(DOWN * 0.3)
        self.play(Create(plane), run_time=1.0)

        origin = plane.c2p(0, 0)
        # dominant harmful write along u1
        harm = Arrow(origin, plane.c2p(3.2, 1.6), buff=0, color=RED, stroke_width=7)
        hlab = txt("harmful write  (σ₁ u₁)", 24, RED).next_to(harm.get_end(), RIGHT, buff=0.15)

        rng = np.random.default_rng(3)
        benign = VGroup()
        for _ in range(9):
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.5, 1.2)
            benign.add(Arrow(origin, plane.c2p(r * np.cos(ang), r * np.sin(ang)),
                             buff=0, color=GREEN, stroke_width=3, max_tip_length_to_length_ratio=0.18))
        blab = txt("benign gain  (many tiny directions)", 24, GREEN).to_corner(DL, buff=0.8)

        self.play(LaggedStart(*[GrowArrow(a) for a in benign], lag_ratio=0.08), FadeIn(blab), run_time=1.2)
        self.play(GrowArrow(harm), FadeIn(hlab), run_time=0.9)
        self.wait(1.3)
        self.geo = VGroup(head, plane, harm, hlab, benign, blab)

    # -- 4. the projection (the repair) ---------------------------------------
    def act_projection(self):
        head2 = txt("Project out the top direction", 34, INK, weight=BOLD).to_edge(UP, buff=0.7)
        plane, harm, hlab, benign, blab = self.geo[1], self.geo[2], self.geo[3], self.geo[4], self.geo[5]
        self.play(FadeOut(self.geo[0], shift=UP * 0.2), FadeIn(head2, shift=DOWN * 0.2), run_time=0.7)

        eq = txt("W_repaired  =  W_base  +  ( dW − σ₁ u₁ v₁ᵀ )", 30, BLUE, weight=BOLD)
        eq.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(eq, shift=UP * 0.2), FadeOut(blab), run_time=0.8)

        # collapse the harmful arrow to zero; keep benign arrows
        self.play(
            harm.animate.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(0.001, 0.001)),
            FadeOut(hlab),
            run_time=1.3,
        )
        gone = txt("harmful write removed", 26, RED, weight=BOLD).move_to(plane.c2p(2.4, 1.4))
        kept = txt("benign gain kept", 26, GREEN, weight=BOLD).move_to(plane.c2p(-2.3, -1.6))
        self.play(FadeIn(gone, scale=1.1), FadeIn(kept), run_time=0.9)
        self.wait(1.4)
        self.play(FadeOut(VGroup(head2, plane, harm, benign, eq, gone, kept)), run_time=0.7)

    # -- 5. result bars -------------------------------------------------------
    def act_result(self):
        head = txt("The result", 34, INK, weight=BOLD).to_edge(UP, buff=0.8)
        self.play(FadeIn(head), run_time=0.6)

        def bar(frac, color, w=1.3, maxh=3.6):
            r = Rectangle(width=w, height=maxh * frac, stroke_width=0).set_fill(color, opacity=1.0)
            return r

        base = -1.9
        # Bengali harm 50% -> 14%
        b_before = bar(0.50, RED)
        b_before.move_to(np.array([-3.4, base + b_before.height / 2, 0]))
        b_after = bar(0.14, RED)
        b_after.move_to(np.array([-1.3, base + b_after.height / 2, 0]))
        lbl_h = txt("Bengali harm", 24, SUB).move_to(np.array([-2.35, base - 0.55, 0]))
        v1 = txt("50%", 26, RED, weight=BOLD).next_to(b_before, UP, buff=0.15)
        v2 = txt("14%", 26, RED, weight=BOLD).next_to(b_after, UP, buff=0.15)

        # MMLU 0.55 -> 0.56 (capability preserved)
        m_before = bar(0.55, GREEN)
        m_before.move_to(np.array([1.6, base + m_before.height / 2, 0]))
        m_after = bar(0.56, GREEN)
        m_after.move_to(np.array([3.7, base + m_after.height / 2, 0]))
        lbl_m = txt("MMLU (capability)", 24, SUB).move_to(np.array([2.65, base - 0.55, 0]))
        w1 = txt("0.55", 26, GREEN, weight=BOLD).next_to(m_before, UP, buff=0.15)
        w2 = txt("0.56", 26, GREEN, weight=BOLD).next_to(m_after, UP, buff=0.15)

        baseline = Line(LEFT * 5.2, RIGHT * 5.2, color=HAIR).move_to(np.array([0, base, 0]))

        self.play(Create(baseline), run_time=0.4)
        self.play(GrowFromEdge(b_before, DOWN), FadeIn(v1), FadeIn(lbl_h), run_time=0.7)
        self.play(TransformFromCopy(b_before, b_after), FadeIn(v2), run_time=0.9)
        self.play(GrowFromEdge(m_before, DOWN), FadeIn(w1), FadeIn(lbl_m), run_time=0.7)
        self.play(TransformFromCopy(m_before, m_after), FadeIn(w2), run_time=0.9)

        caption = txt("Harm cut 50% to 14%. Capability untouched. No retraining.", 28, INK, weight=BOLD)
        caption.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.9)
        self.wait(1.6)
        self.play(FadeOut(VGroup(head, baseline, b_before, b_after, m_before, m_after,
                                 v1, v2, w1, w2, lbl_h, lbl_m, caption)), run_time=0.7)

    # -- end ------------------------------------------------------------------
    def end_card(self):
        t1 = txt("Repair beats prevention.", 44, INK, weight=BOLD)
        t2 = txt("Post-hoc. Training-free. Low-rank.", 30, SUB)
        url = txt("getalife-lexsi-study.vercel.app", 26, BLUE)
        g = VGroup(t1, t2).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        url.next_to(g, DOWN, buff=0.7)
        self.play(Write(t1), run_time=1.0)
        self.play(FadeIn(t2, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(url), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(VGroup(g, url)), run_time=0.8)
