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
    from matplotlib import colormaps
    norm = np.clip((grid.T-low) / max(high-low, 1e-30), 0, 1)
    rgba = (colormaps[cmap](norm)*255).astype(np.uint8)
    # Image row zero is at the top; stored x2=0 must be at the bottom.
    obj = ImageMobject(np.flipud(rgba)).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    obj.set_height(size).move_to(center)
    border = Square(side_length=size, stroke_color=DIM, stroke_width=1).move_to(center)
    # A real colour ramp and numeric limits, not colour without units.
    ramp = (colormaps[cmap](np.linspace(1, 0, 256))*255).astype(np.uint8)[:, None, :]
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
