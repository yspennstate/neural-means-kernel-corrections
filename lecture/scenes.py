"""Explicit LaTeX boards and animated scientific geometry.

Set NMKC_CHAPTER=01. Render LectureChapter after recording its narration.
Set NMKC_BOARD to a board key and render BoardPreview with -s for a still.
No equation fallback: a typesetting error must fail the build.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from manim import *

from audio_contract import validate

HERE = Path(__file__).resolve().parent
CHAPTER = os.environ.get("NMKC_CHAPTER", "01")
DATA = json.loads((HERE / "chapters" / (CHAPTER + ".json")).read_text(encoding="utf-8"))
BG, INK, DIM = "#111A23", "#F2EEE7", "#AEBBC9"
GOLD, BLUE, GREEN, RED = "#E5B85C", "#70B7DF", "#94C9A9", "#EC8C86"
config.background_color = BG
config.max_inflight_encoders = 1
config.tex_dir = str(HERE / "media" / "Tex" / f"p{os.getpid()}")
config.text_dir = str(HERE / "media" / "texts" / f"p{os.getpid()}")
TEMPLATE = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")
TEMPLATE.add_to_preamble(r"\usepackage[T1]{fontenc}\usepackage{amsmath,amssymb,bm,newtxtext,newtxmath}")


def escape(text):
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(table.get(c, c) for c in text)


def prose(text, size=31, color=INK, width=None):
    body = escape(text)
    if width:
        body = rf"\parbox{{{width}cm}}{{\raggedright {body}}}"
    return Tex(body, tex_template=TEMPLATE, font_size=size, color=color)


def formula(text, size=35, color=GOLD, max_width=5.0):
    obj = MathTex(text, tex_template=TEMPLATE, font_size=size, color=color)
    if obj.width > max_width:
        factor = max_width / obj.width
        if factor < .56:
            raise ValueError(f"Equation too dense for the board: {text}")
        obj.scale(factor)
    return obj


def small_label(text, point, size=25, color=DIM):
    return prose(text, size=size, color=color).move_to(point)


def axes(x=(0, 1, .5), y=(0, 1, .5), width=3.0, height=2.6):
    return Axes(x_range=x, y_range=y, x_length=width, y_length=height,
                tips=False, axis_config={"color": DIM, "stroke_width": 1.5, "include_ticks": True})


def mechanics():
    p = HERE / "assets/mechanics_examples.npz"
    receipt = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(p.read_bytes()).hexdigest() != receipt["npz_sha256"]:
        raise ValueError("Mechanics examples do not match their source receipt")
    return np.load(p), receipt


def field_image(grid, cmap, low, high, center, size=2.15):
    palette_path = HERE / "assets/colormaps.json"
    palette = np.asarray(json.loads(palette_path.read_text())[cmap], dtype=np.uint8)
    norm = np.clip((grid.T-low) / max(high-low, 1e-30), 0, 1)
    rgba = palette[np.rint(norm*255).astype(int)]
    # Image row zero is at the top; stored x2=0 must be at the bottom.
    obj = ImageMobject(np.flipud(rgba)).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    obj.set_height(size).move_to(center)
    border = Square(side_length=size, stroke_color=DIM, stroke_width=1).move_to(center)
    # A real colour ramp and numeric limits, not colour without units.
    ramp = palette[::-1, None, :]
    bar = ImageMobject(ramp).stretch_to_fit_height(size).stretch_to_fit_width(.13)
    bar.next_to(border, RIGHT, buff=.1)
    hi = formula(f"{high:.3g}", size=17, color=DIM).next_to(bar, UP, buff=.06)
    lo = formula(f"{low:.3g}", size=17, color=DIM).next_to(bar, DOWN, buff=.06)
    return Group(obj, border, bar, hi, lo)


def build_visual(spec):
    kind = spec["kind"]
    if kind == "elastic_domain":
        square = Square(3.0, color=BLUE, fill_color=BLUE, fill_opacity=.09).move_to([2.5, .0, 0])
        fibre = Circle(.72, color=GOLD, fill_color=GOLD, fill_opacity=.22).move_to(square)
        clamps = VGroup(*(Line([1+i*.25, -1.5, 0], [.85+i*.25, -1.72, 0], color=DIM)
                          for i in range(13)))
        arrows = VGroup(*(Arrow([1.15+i*.45, 1.5, 0], [1.15+i*.45, 2.0+.35*np.sin(i), 0],
                               buff=0, color=GOLD, stroke_width=3, max_tip_length_to_length_ratio=.25)
                          for i in range(7)))
        labels = VGroup(small_label("Fibre", [2.5, 0, 0], 28, GOLD),
                        small_label("Matrix", [3.5, -.95, 0], 25, BLUE),
                        small_label("Clamped boundary", [2.5, -2.08, 0]),
                        formula(r"\bar t(x_1)", 29).move_to([2.5, 2.58, 0]))
        group = VGroup(square, fibre, clamps, arrows, labels)
        return group, [lambda: Indicate(fibre, color=GOLD), lambda: LaggedStart(*(Indicate(a) for a in arrows), lag_ratio=.08),
                       lambda: Indicate(square, color=BLUE)]
    if kind in ("field_pair", "field_error"):
        data, receipt = mechanics()
        case = spec["case"]
        load, target, prediction = (data[f"{case}_{k}"] for k in ("load", "target", "prediction"))
        error = np.abs(prediction-target)
        if kind == "field_pair":
            ymin, ymax = float(load.min()), float(load.max())
            pad = max((ymax-ymin)*.13, 1)
            ax = axes(y=(ymin-pad, ymax+pad, max((ymax-ymin)/2, 1)), width=2.55, height=2.5).move_to([.75, .25, 0])
            points = [ax.c2p(x, y) for x, y in zip(np.linspace(0, 1, 41), load)]
            curve = VMobject(color=GOLD, stroke_width=3).set_points_as_corners(points)
            labels = VGroup(small_label("Boundary load", [.7, 2.0, 0], 27),
                            formula(r"x_1", 24, DIM).next_to(ax, DOWN, buff=.12),
                            formula(f"{ymin:.1f}", 18, DIM).next_to(ax.c2p(0, ymin), LEFT, buff=.1),
                            formula(f"{ymax:.1f}", 18, DIM).next_to(ax.c2p(0, ymax), LEFT, buff=.1))
            image = field_image(target, "viridis", float(target.min()), float(target.max()), [4.55, .25, 0], 2.5)
            ilabel = small_label("Reference stress", [4.55, 2.0, 0], 27)
            arrow = Arrow([2.35, .3, 0], [2.95, .3, 0], color=INK, buff=0)
            group = Group(ax, curve, labels, image, ilabel, arrow)
            return group, [lambda: ShowPassingFlash(curve.copy().set_color(INK), time_width=.4),
                           lambda: Indicate(image[1], color=GOLD), lambda: Indicate(arrow)]
        low, high = min(float(target.min()), float(prediction.min())), max(float(target.max()), float(prediction.max()))
        values = [(target, "Reference", "viridis", low, high),
                  (prediction, "Prediction", "viridis", low, high),
                  (error, "Absolute error", "magma", 0., float(error.max()))]
        images = []; parts = []
        for center, (grid, label, cmap, lo, hi) in zip(([.1, .3, 0], [2.6, .3, 0], [5.1, .3, 0]), values):
            obj = field_image(grid, cmap, lo, hi, center, 1.9); images.append(obj); parts.append(obj)
            parts.append(small_label(label, [center[0], 1.75, 0], 24))
        score = receipt["cases"][case]["relative_error"]*100
        parts.append(formula(rf"\ell={score:.3f}\%", 32).move_to([2.6, -1.5, 0]))
        parts.append(small_label("Reference and prediction share a colour scale", [2.6, -2.1, 0], 21))
        return Group(*parts), [lambda: Indicate(images[0][1]), lambda: Indicate(images[2][1], color=RED),
                               lambda: Indicate(images[1][1], color=BLUE)]
    if kind == "reflection_pair":
        data, receipt = mechanics()
        case = spec.get("case", "median")
        field = data[case+"_target"]
        low, high = float(field.min()), float(field.max())
        original = field_image(field, "viridis", low, high, [.65, .4, 0], 2.4)
        reflected = field_image(np.flip(field, axis=0), "viridis", low, high, [4.25, .4, 0], 2.4)
        arrow = DoubleArrow([2.05, .4, 0], [2.85, .4, 0], color=GOLD, buff=.05)
        labels = VGroup(small_label("Original coordinates", [.65, 2.05, 0], 25),
                        small_label("Reflected coordinates", [4.25, 2.05, 0], 25),
                        formula(r"x_1\mapsto1-x_1", 30, GOLD).move_to([2.6, -1.5, 0]),
                        small_label("Transform back before averaging predictions", [2.6, -2.2, 0], 23))
        group = Group(original, reflected, arrow, labels)
        return group, [lambda: Indicate(original[1], color=BLUE), lambda: Indicate(reflected[1], color=GOLD),
                       lambda: Indicate(arrow, color=GOLD)]
    if kind == "pipeline_stages":
        def box(label, x, y, color):
            frame = RoundedRectangle(width=1.95, height=.84, corner_radius=.08,
                                     color=color, fill_color=color, fill_opacity=.1).move_to([x,y,0])
            text = formula(label, 28, color, 1.75).move_to(frame)
            return VGroup(frame, text)
        mean = box(r"m(u)", .1, 1.25, BLUE)
        target = box(r"G(u)", 4.8, 1.25, INK)
        residual = box(r"G-m", 2.45, -.05, GOLD)
        kernel = box(r"\widehat r(u)", 2.45, -1.65, GREEN)
        final = box(r"m+\widehat r", 4.8, -1.65, INK)
        links = VGroup(Arrow(mean.get_right(), residual.get_left(), buff=.08, color=BLUE),
                       Arrow(target.get_left(), residual.get_right(), buff=.08, color=GOLD),
                       Arrow(residual.get_bottom(), kernel.get_top(), buff=.08, color=GREEN),
                       Arrow(kernel.get_right(), final.get_left(), buff=.08, color=GREEN))
        bypass = VMobject(color=BLUE, stroke_width=2).set_points_as_corners(
            [mean.get_bottom(), [.1,-2.55,0], [4.8,-2.55,0], final.get_bottom()])
        labels = VGroup(small_label("Neural mean", [.1,2.05,0], 24, BLUE),
                        small_label("Reference", [4.8,2.05,0], 24),
                        small_label("Fit a kernel", [1.0,-.9,0], 22, GREEN))
        group = VGroup(mean, target, residual, kernel, final, links, bypass, labels)
        return group, [lambda: Indicate(mean, color=BLUE), lambda: Indicate(residual, color=GOLD),
                       lambda: Indicate(final, color=GREEN)]
    if kind == "data_split":
        top = Rectangle(width=6.4, height=.85, color=INK).move_to([2.7, 1.45, 0])
        dev = Rectangle(width=3.2, height=.85, color=BLUE, fill_color=BLUE, fill_opacity=.3).align_to(top, LEFT)
        dev.set_y(top.get_y())
        test = Rectangle(width=3.2, height=.85, color=GOLD, fill_color=GOLD, fill_opacity=.25).align_to(top, RIGHT)
        test.set_y(top.get_y())
        texts = VGroup(small_label("20,000 development", dev.get_center(), 24),
                       small_label("20,000 public test", test.get_center(), 24))
        train = Rectangle(width=5.32, height=.85, color=BLUE, fill_color=BLUE, fill_opacity=.3).move_to([2.1, -.3, 0])
        val = Rectangle(width=.8, height=.85, color=GREEN, fill_color=GREEN, fill_opacity=.3).next_to(train, RIGHT, buff=.15)
        labels = VGroup(small_label("19,000 training", train.get_center(), 26),
                        small_label("1,000 validation", [val.get_x(), -1.05, 0], 23),
                        small_label("Validation width enlarged for readability", [2.7, -1.8, 0], 20))
        links = VGroup(Arrow(dev.get_bottom(), train.get_top(), buff=.1, color=BLUE),
                       Arrow(dev.get_bottom(), val.get_top(), buff=.1, color=GREEN))
        group = VGroup(top, dev, test, texts, train, val, labels, links)
        return group, [lambda: Indicate(top), lambda: Indicate(val, color=GREEN), lambda: Indicate(test, color=GOLD)]
    if kind == "two_kernel_roles":
        def box(text, point, color):
            rect = RoundedRectangle(width=2.55, height=1.0, corner_radius=.1, color=color,
                                    fill_color=color, fill_opacity=.1).move_to(point)
            label = prose(text, size=26, color=INK).move_to(rect)
            if label.width > 2.3: label.scale(2.3/label.width)
            return VGroup(rect, label)
        mean = box("Neural mean", [.8, 1.0, 0], BLUE)
        residual = box("Residual kernel", [4.35, 1.0, 0], GOLD)
        feature = box("Frozen features", [.8, -1.0, 0], BLUE)
        head = box("Feature kernel", [4.35, -1.0, 0], GREEN)
        links = VGroup(Arrow(mean.get_right(), residual.get_left(), buff=.1, color=GOLD),
                       Arrow(feature.get_right(), head.get_left(), buff=.1, color=GREEN))
        group = VGroup(mean, residual, feature, head, links)
        return group, [lambda: Indicate(residual), lambda: Indicate(feature), lambda: Indicate(links)]
    if kind == "residual_blocks":
        # A component diagram, not a false claim that all depicted directions
        # are mutually orthogonal in the two-dimensional screen plane.
        common = RoundedRectangle(width=5.8, height=.72, color=GOLD, fill_color=GOLD,
                                  fill_opacity=.22, corner_radius=.08).move_to([2.6, 1.65, 0])
        common_label = prose("Common to every predictor", 26, GOLD).move_to(common)
        archs = VGroup(); seeds = VGroup(); arrows = VGroup()
        for a, x in enumerate((.55, 2.6, 4.65)):
            arch = RoundedRectangle(width=1.7, height=.72, color=BLUE, fill_color=BLUE,
                                    fill_opacity=.18, corner_radius=.07).move_to([x, .05, 0])
            label = formula(rf"b_{{{a+1}}}", 30, BLUE).move_to(arch)
            archs.add(VGroup(arch, label))
            arrows.add(Arrow([x, 1.25, 0], [x, .5, 0], buff=.08, color=DIM))
            for k, dx in enumerate((-.57, -.19, .19, .57)):
                dot = Dot([x+dx, -1.35, 0], radius=.095, color=GREEN)
                seeds.add(dot)
                arrows.add(Line([x, -.35, 0], dot.get_top(), color=DIM, stroke_width=1.2))
        labels = VGroup(small_label("Architecture components", [2.6, -.58, 0], 23, BLUE),
                        small_label("Individual seed components", [2.6, -1.93, 0], 23, GREEN),
                        small_label("Orthogonality is in the model space, not the drawing", [2.6, -2.5, 0], 18))
        group = VGroup(common, common_label, archs, seeds, arrows, labels)
        return group, [lambda: Indicate(common, color=GOLD), lambda: Indicate(archs, color=BLUE),
                       lambda: Indicate(seeds, color=GREEN)]
    if kind == "block_matrix":
        M, K = spec["M"], spec["K"]; n = M*K
        rw, rb = spec["rho_w"], spec["rho_b"]
        cell = 3.95/n
        cells = VGroup(); diag = VGroup(); within = VGroup(); between = VGroup()
        for i in range(n):
            for j in range(n):
                value = 1 if i == j else rw if i//K == j//K else rb
                color = INK if i == j else GOLD if i//K == j//K else BLUE
                square = Square(cell, color=BG, stroke_width=.55, fill_color=color,
                                fill_opacity=.35+.6*value).move_to(
                                    [2.6+(j-(n-1)/2)*cell, .3+((n-1)/2-i)*cell, 0])
                cells.add(square)
                (diag if i == j else within if i//K == j//K else between).add(square)
        boundaries = VGroup()
        for a in range(M):
            boundaries.add(Square(cell*K, color=GOLD, stroke_width=1.8).move_to(
                [2.6+(a*K+(K-1)/2-(n-1)/2)*cell,
                 .3+((n-1)/2-a*K-(K-1)/2)*cell, 0]))
        labels = VGroup(formula(r"S/\bar e^{\,2}", 28, INK).move_to([2.6, 2.58, 0]),
                        small_label("Diagonal 1", [.2, -2.15, 0], 21, INK),
                        small_label(f"Within {rw:g}", [2.5, -2.15, 0], 21, GOLD),
                        small_label(f"Between {rb:g}", [4.9, -2.15, 0], 21, BLUE))
        group = VGroup(cells, boundaries, labels)
        return group, [lambda: Indicate(diag, color=INK, scale_factor=1.01),
                       lambda: Indicate(within, color=GOLD, scale_factor=1.01),
                       lambda: Indicate(between, color=BLUE, scale_factor=1.01)]
    if kind == "architecture_weights":
        weights = [[.25, .1, .1, .1], [.05, .1, .1, .05], [.05, .04, .03, .03]]
        colors = [BLUE, GOLD, GREEN, RED]
        def bars(values):
            group = VGroup()
            for a, (x, ws) in enumerate(zip((.4, 2.6, 4.8), values)):
                bottom = -1.7
                for k, w in enumerate(ws):
                    h = w*6
                    part = Rectangle(width=.95, height=h, color=BG, stroke_width=1.5,
                                     fill_color=colors[k], fill_opacity=.8).move_to([x, bottom+h/2, 0])
                    group.add(part); bottom += h
                group.add(formula(rf"W_{a+1}={sum(ws):.3f}", 24, INK).move_to([x, 2.0, 0]))
                group.add(small_label(f"Architecture {a+1}", [x, -2.15, 0], 21))
            return group
        original = bars(weights); uniform = bars([[1/12]*4 for _ in range(3)])
        baseline = Line([-.3, -1.72, 0], [5.5, -1.72, 0], color=DIM, stroke_width=1)
        label = small_label("Coloured pieces are individual seed weights", [2.6, -2.6, 0], 19)
        group = VGroup(original, baseline, label)
        return group, [lambda: Indicate(original, scale_factor=1.02),
                       lambda: Indicate(original[0], color=BLUE),
                       lambda: Transform(original, uniform)]
    if kind == "floor_components":
        M, K, rw, rb = spec["M"], spec["K"], spec["rho_w"], spec["rho_b"]
        values = [rb, (rw-rb)/M, (1-rw)/(M*K)]
        expressions = [r"\varrho_b", r"(\varrho_w-\varrho_b)/M", r"(1-\varrho_w)/(MK)"]
        colors = [GOLD, BLUE, GREEN]
        bars = VGroup(); labels = VGroup()
        for y, value, expression, color in zip((1.4, -.1, -1.6), values, expressions, colors):
            bar = Rectangle(width=6*value, height=.4, color=color, fill_color=color,
                            fill_opacity=.7).move_to([-.2+3*value, y, 0])
            bars.add(bar)
            labels.add(formula(expression+rf"={value:.4f}", 27, color, max_width=6.5).move_to([2.6, y+.55, 0]))
        group = VGroup(bars, labels)
        return group, [lambda: Indicate(bars[0]), lambda: Indicate(bars[1]), lambda: Indicate(bars[2])]
    if kind == "block_floor_curve":
        M, rw, rb = spec["M"], spec["rho_w"], spec["rho_b"]
        low = rb+(rw-rb)/M
        ax = axes(x=(1, 61, 20), y=(.64, .8, .04), width=5.5, height=3.35).move_to([2.6, .3, 0])
        curve = ax.plot(lambda k: low+(1-rw)/(M*k), x_range=[1, 60], color=GOLD, stroke_width=3)
        asymptote = DashedLine(ax.c2p(1, low), ax.c2p(60, low), color=BLUE, stroke_width=2)
        labels = VGroup(prose("Seeds per architecture", 24).move_to([2.6, -2.15, 0]),
                        prose("Normalized squared-error lower bound", 22).move_to([2.6, 2.35, 0]),
                        formula(rf"\varrho_b+(\varrho_w-\varrho_b)/M={low:.4f}", 24, BLUE, 6.5).move_to([2.6, -2.75, 0]))
        for x in (1, 20, 40, 60):
            labels.add(formula(str(x), 18, DIM).next_to(ax.c2p(x, .64), DOWN, buff=.13))
        for y in (.64, .68, .72, .76, .8):
            labels.add(formula(f"{y:.2f}", 18, DIM).next_to(ax.c2p(1, y), LEFT, buff=.13))
        dot = Dot(ax.c2p(1, low+(1-rw)/M), color=INK, radius=.065)
        group = VGroup(ax, curve, asymptote, labels, dot)
        return group, [lambda: MoveAlongPath(dot, curve), lambda: Indicate(asymptote, color=BLUE),
                       lambda: Indicate(curve, color=GOLD)]
    if kind == "kernel_projection":
        from kernel_geometry import geometry
        ridge = spec.get("ridge", 0.)
        result = geometry([[1, 0]], [3, 2], ridge)
        coefficient = result["coefficients"][0]
        ax = axes(x=(-.3, 4, 1), y=(-.3, 3, 1), width=5.3, height=3.9).move_to([2.7, .15, 0])
        origin = ax.c2p(0, 0)
        train = Arrow(origin, ax.c2p(1, 0), buff=0, color=BLUE, stroke_width=5)
        query = Arrow(origin, ax.c2p(3, 2), buff=0, color=GOLD, stroke_width=4)
        fitted = Arrow(origin, ax.c2p(coefficient, 0), buff=0, color=GREEN, stroke_width=4)
        gap = Arrow(ax.c2p(coefficient, 0), ax.c2p(3, 2), buff=0, color=RED, stroke_width=4)
        projection = DashedLine(ax.c2p(3, 0), ax.c2p(3, 2), color=DIM, stroke_width=1.5)
        labels = VGroup(formula(r"\phi(u)=(3,2)", 25, GOLD).next_to(query.get_end(), UP, buff=.15),
                        formula(r"\phi(x_1)=(1,0)", 24, BLUE).next_to(ax.c2p(1, 0), DOWN, buff=.23),
                        formula(rf"c\phi(x_1)=({coefficient:g},0)", 23, GREEN).move_to([3.65, -2.05, 0]),
                        formula(r"g_u", 30, RED).next_to(gap.get_center(), RIGHT, buff=.12),
                        formula(rf"\|g_u\|^2={result['exact_squared']:g}", 29, RED).move_to([2.6, 2.5, 0]),
                        small_label("The horizontal line is the observed span", [2.6, -2.62, 0], 21))
        group = VGroup(ax, projection, train, fitted, query, gap, labels)
        return group, [lambda: Indicate(query, color=GOLD),
                       lambda: Indicate(fitted, color=GREEN), lambda: Indicate(gap, color=RED)]
    if kind == "cauchy_ball":
        center = np.array([2.5, .05, 0]); radius = 1.8
        ball = Circle(radius, color=BLUE, fill_color=BLUE, fill_opacity=.07).move_to(center)
        g = np.array([1., 2., 0]); direction = g/np.linalg.norm(g)
        fixed = Arrow(center, center+2.4*direction, buff=0, color=RED, stroke_width=4)
        theta = .2
        r = Arrow(center, center+radius*np.array([np.cos(theta), np.sin(theta), 0]),
                  buff=0, color=GOLD, stroke_width=5)
        projection_point = center+radius*np.dot([np.cos(theta), np.sin(theta), 0], direction)*direction
        projection = DashedLine(r.get_end(), projection_point, color=DIM, stroke_width=2)
        target = Arrow(center, center+radius*direction, buff=0, color=GOLD, stroke_width=5)
        labels = VGroup(formula(r"\|r\|=\rho", 29, BLUE).move_to([4.8, -.7, 0]),
                        formula(r"g_u", 30, RED).next_to(fixed.get_end(), UP, buff=.1),
                        formula(r"r", 30, GOLD).next_to(r.get_end(), RIGHT, buff=.1),
                        formula(r"|\langle r,g_u\rangle|=\rho\|g_u\|\,|\cos\theta|", 27, INK, 6.2).move_to([2.6, -2.4, 0]))
        group = VGroup(ball, fixed, r, projection, labels)
        return group, [lambda: Indicate(projection), lambda: Indicate(ball, color=BLUE),
                       lambda: AnimationGroup(Transform(r, target), FadeOut(projection))]
    if kind == "invisible_residuals":
        ax = axes(x=(-1, 3.5, 1), y=(-2, 2.5, 1), width=5.3, height=3.8).move_to([2.6, .05, 0])
        origin = ax.c2p(0, 0)
        train = Arrow(origin, ax.c2p(1, 0), buff=0, color=BLUE, stroke_width=5)
        plus = Arrow(origin, ax.c2p(0, 1.6), buff=0, color=GOLD, stroke_width=4)
        minus = Arrow(origin, ax.c2p(0, -1.6), buff=0, color=GREEN, stroke_width=4)
        query = Arrow(origin, ax.c2p(3, 2), buff=0, color=DIM, stroke_width=2)
        zero = Dot(origin, radius=.08, color=RED)
        labels = VGroup(formula(r"r^+", 29, GOLD).next_to(plus.get_end(), LEFT, buff=.15),
                        formula(r"r^-", 29, GREEN).next_to(minus.get_end(), LEFT, buff=.15),
                        formula(r"\phi(x_1)", 26, BLUE).next_to(train.get_end(), DOWN, buff=.2),
                        formula(r"\phi(u)", 27, DIM).next_to(query.get_end(), UP, buff=.1),
                        formula(r"\langle r^\pm,\phi(x_1)\rangle=0", 27, INK, 6.2).move_to([2.6, -2.55, 0]))
        group = VGroup(ax, query, train, plus, minus, zero, labels)
        return group, [lambda: Indicate(train, color=BLUE), lambda: Indicate(zero, color=RED),
                       lambda: AnimationGroup(Indicate(plus, color=GOLD), Indicate(minus, color=GREEN))]
    if kind == "minimax_interval":
        left, right = np.array([-.05, .7, 0]), np.array([5.15, .7, 0])
        mid = (left+right)/2; prediction = mid+np.array([.9, 0, 0])
        line = Line(left-np.array([.3, 0, 0]), right+np.array([.3, 0, 0]), color=DIM)
        a, b = Dot(left, radius=.1, color=GREEN), Dot(right, radius=.1, color=GOLD)
        psi = Dot(prediction, radius=.095, color=RED)
        dist_left = DoubleArrow(left+[0, -.65, 0], prediction+[0, -.65, 0], buff=.04, color=GREEN)
        dist_right = DoubleArrow(prediction+[0, .65, 0], right+[0, .65, 0], buff=.04, color=GOLD)
        total = DoubleArrow(left+[0, -1.4, 0], right+[0, -1.4, 0], buff=.04, color=INK)
        labels = VGroup(formula(r"-\rho P_0", 27, GREEN).next_to(a, UP, buff=.2),
                        formula(r"+\rho P_0", 27, GOLD).next_to(b, UP, buff=.2),
                        formula(r"\psi", 31, RED).next_to(psi, DOWN, buff=.15),
                        formula(r"2\rho P_0", 28, INK).next_to(total, DOWN, buff=.18),
                        small_label("At least one error is half the separation", [2.6, -2.25, 0], 25))
        group = VGroup(line, a, b, psi, dist_left, dist_right, total, labels)
        return group, [lambda: Indicate(psi, color=RED), lambda: Indicate(total),
                       lambda: AnimationGroup(psi.animate.move_to(mid),
                           Transform(dist_left, DoubleArrow(left+[0,-.65,0], mid+[0,-.65,0], buff=.04, color=GREEN)),
                           Transform(dist_right, DoubleArrow(mid+[0,.65,0], right+[0,.65,0], buff=.04, color=GOLD)))]
    if kind == "nugget_factors":
        ax = axes(x=(0, 2, .5), y=(0, 11, 2), width=5.4, height=3.5).move_to([2.6, .3, 0])
        # For phi(x)=(1,0), phi(u)=(3,2), n=1:
        interpolation = ax.plot(lambda l: 4., x_range=[0, 2], color=BLUE, stroke_width=3)
        exact = ax.plot(lambda l: 4+9*(l/(1+l))**2, x_range=[0, 2], color=GOLD, stroke_width=3)
        posterior = ax.plot(lambda l: 13-9/(1+l), x_range=[0, 2], color=GREEN, stroke_width=3)
        labels = VGroup(formula(r"\lambda", 27, INK).next_to(ax, DOWN, buff=.32),
                        small_label("Squared geometric factors", [2.6, 2.5, 0], 27),
                        formula(r"P_0^2", 25, BLUE).move_to([.0, -2.35, 0]),
                        formula(r"\widetilde P_\lambda^2", 25, GOLD).move_to([2.3, -2.35, 0]),
                        formula(r"P_\lambda^2", 25, GREEN).move_to([4.8, -2.35, 0]))
        for x in (0, .5, 1, 1.5, 2):
            labels.add(formula(f"{x:g}", 19, DIM).next_to(ax.c2p(x, 0), DOWN, buff=.1))
        for y in (0, 2, 4, 6, 8, 10):
            labels.add(formula(str(y), 19, DIM).next_to(ax.c2p(0, y), LEFT, buff=.1))
        point = Dot(ax.c2p(.5, 5), color=INK, radius=.065)
        group = VGroup(ax, interpolation, exact, posterior, labels, point)
        return group, [lambda: Indicate(exact, color=GOLD), lambda: Indicate(posterior, color=GREEN),
                       lambda: Indicate(interpolation, color=BLUE)]
    raise ValueError(f"Unimplemented visual: {kind}")


class BoardMixin:
    def frame(self, board):
        title = prose(board["title"], size=41)
        if title.width > 12.9: title.scale(12.9/title.width)
        title.move_to([0, 3.32, 0])
        rule = Line([-6.5, 2.8, 0], [6.5, 2.8, 0], color=DIM, stroke_width=1)
        divider = Line([-1.15, 2.5, 0], [-1.15, -2.8, 0], color=DIM, stroke_width=.7)
        label = prose(board["visual"]["label"], size=20)
        if label.width > 12.3: label.scale(12.3/label.width)
        label.move_to([0, -3.35, 0])
        number = prose(f"Chapter {int(CHAPTER)}", size=18, color=DIM).move_to([-5.7, -3.78, 0])
        group, effects = build_visual(board["visual"])
        self.add(title, rule, divider, label, number, group)
        return group, effects

    def segment_text(self, segment):
        line = prose(segment["line"], size=31, width=6.0)
        if line.width > 4.75: line.scale(4.75/line.width)
        line.move_to([-3.85, 1.0, 0])
        equation = formula(segment["math"], size=35, max_width=4.8).move_to([-3.85, -.45, 0])
        return VGroup(line, equation)


class BoardPreview(BoardMixin, Scene):
    def construct(self):
        key = os.environ.get("NMKC_BOARD", DATA["boards"][0]["key"])
        board = next(b for b in DATA["boards"] if b["key"] == key)
        self.frame(board)
        index = int(os.environ.get("NMKC_SEGMENT", "0"))
        self.add(self.segment_text(board["segments"][index]))


class BoardSamples(BoardMixin, Scene):
    def construct(self):
        out = HERE / "media" / "board_samples" / CHAPTER
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for board in DATA["boards"]:
            for i, segment in enumerate(board["segments"], 1):
                self.clear()
                self.frame(board)
                self.add(self.segment_text(segment))
                self.renderer.update_frame(self)
                path = out / f'{board["key"]}_{i:02}.png'
                self.renderer.camera.get_image().save(path)
                rows.append({"board": board["key"], "segment": i, "path": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (out / "manifest.json").write_text(json.dumps({
            "kind": "author_layout_samples", "chapter": CHAPTER,
            "human_visual_review": "PENDING", "rows": rows}, indent=2), encoding="utf-8")


class LectureChapter(BoardMixin, Scene):
    def construct(self):
        timing = []
        # Fail before spending render time if any recording is stale or missing.
        durations = {}
        for board in DATA["boards"]:
            for i, segment in enumerate(board["segments"], 1):
                key = f'{board["key"]}_{i:02}'
                durations[key] = validate(HERE / "audio" / CHAPTER / (key + ".wav"), segment["say"])
        for board in DATA["boards"]:
            self.clear()
            _, effects = self.frame(board)
            text = None
            for i, segment in enumerate(board["segments"], 1):
                key = f'{board["key"]}_{i:02}'
                start = self.renderer.time
                current = self.segment_text(segment)
                if text is None:
                    self.play(FadeIn(current), run_time=.45)
                else:
                    self.play(ReplacementTransform(text, current), run_time=.45)
                text = current
                # All voice duration remains available; animation never truncates speech.
                self.add_sound(str(HERE / "audio" / CHAPTER / (key + ".wav")))
                voice_start = self.renderer.time
                duration = durations[key]
                self.wait(max(.05, duration*.22))
                self.play(effects[min(i-1, len(effects)-1)](), run_time=1.4)
                remaining = duration - (self.renderer.time-voice_start)
                if remaining < -.02:
                    raise ValueError(f"Animation outlasts narration: {key}")
                self.wait(max(0, remaining))
                hold = .7 if CHAPTER == "01" else 1.6
                self.wait(hold)
                timing.append({"key": key, "start": start, "voice_start": voice_start,
                               "voice_seconds": duration, "end": self.renderer.time,
                               "board": board["key"]})
        out = HERE / "timing"
        out.mkdir(exist_ok=True)
        (out / f"chapter{CHAPTER}.json").write_text(json.dumps({
            "chapter": CHAPTER, "script_sha256": hashlib.sha256(
                (HERE/"chapters"/(CHAPTER+".json")).read_bytes()).hexdigest(),
            "segments": timing, "seconds": self.renderer.time}, indent=2), encoding="utf-8")
