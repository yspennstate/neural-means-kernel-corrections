"""Explicit LaTeX boards and animated scientific geometry.

Set NMKC_CHAPTER=01. Render LectureChapter after recording its narration.
Set NMKC_BOARD to a board key and render BoardPreview with -s for a still.
No equation fallback: a typesetting error must fail the build.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

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
# Each partial has a movie, command log and receipt. Preserve all three so
# Manim's cache limit cannot silently remove the production evidence.
config.max_files_cached = -1
from encoder_policy import record as record_encoder_policy
ENCODER_POLICY = record_encoder_policy(HERE)
config.tex_dir = str(HERE / "media" / "Tex" / f"p{os.getpid()}")
if (HERE / "tex_seed").exists():
    frozen_inputs = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    Path(config.tex_dir).mkdir(parents=True, exist_ok=True)
    for asset in (HERE / "tex_seed").iterdir():
        key = asset.relative_to(HERE).as_posix()
        if (asset.suffix not in (".tex", ".svg")
                or hashlib.sha256(asset.read_bytes()).hexdigest() != frozen_inputs.get(key)):
            raise ValueError("Frozen TeX vector asset identity failed")
        shutil.copyfile(asset, Path(config.tex_dir) / asset.name)
config.text_dir = str(HERE / "media" / "texts" / f"p{os.getpid()}")
TEMPLATE = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")
from tex_style import PREAMBLE
TEMPLATE.add_to_preamble(PREAMBLE)
from tex_rules import install as install_tex_rules
install_tex_rules(HERE)


def escape(text):
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(table.get(c, c) for c in text)


def prose(text, size=31, color=INK, width=None):
    body = escape(text)
    if width:
        body = rf"\parbox{{{width}cm}}{{\raggedright {body}}}"
    return Tex(body, tex_template=TEMPLATE, font_size=size, color=color)


def formula(text, size=35, color=GOLD, max_width=5.0, shrink=True):
    obj = MathTex(text, tex_template=TEMPLATE, font_size=size, color=color)
    if obj.width > max_width:
        if not shrink:
            raise ValueError(f"Break this equation into readable lines: width {obj.width:.3f} > {max_width}: {text}")
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
    palette = np.asarray(json.loads(palette_path.read_text(encoding="utf-8"))[cmap], dtype=np.uint8)
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
    if kind == "sensitivity_results":
        receipt=json.loads((HERE/'assets/sensitivity_figures.json').read_text(encoding='utf-8'))
        row=next(r for r in receipt['rows'] if r['kind']==spec['chart'])
        path=HERE/'assets'/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Sensitivity plot differs from its evidence receipt')
        plot=ImageMobject(str(path)).set_width(7.1).move_to([2.65,0,0])
        if spec['chart']=='centering':
            boxes=[Rectangle(width=6.6,height=2.2,color=BLUE).move_to([2.7,1.25,0]),
                   Rectangle(width=6.6,height=2.2,color=BLUE).move_to([2.7,1.25,0]),
                   Rectangle(width=6.6,height=2.2,color=GOLD).move_to([2.7,-1.3,0])]
        else:
            boxes=[SurroundingRectangle(plot,color=DIM,buff=.05),
                   Rectangle(width=6.2,height=4.5,color=GOLD).move_to([2.9,0,0]),
                   Rectangle(width=1.65,height=4.4,color=GREEN).move_to([5.0,-.1,0])]
        return Group(plot),[lambda box=box:ShowPassingFlash(box) for box in boxes]
    if kind == "uq_measured":
        receipt=json.loads((HERE/'assets/uq_figures.json').read_text(encoding='utf-8'))
        row=next(r for r in receipt['rows'] if r['kind']==spec['figure'])
        path=HERE/'assets'/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('UQ plot differs from its receipt')
        plot=ImageMobject(str(path)).set_width(7.1).move_to([2.65,.0,0])
        top=Rectangle(width=6.45,height=2.15,color=GREEN).move_to([2.9,1.45,0])
        bottom=Rectangle(width=6.45,height=2.1,color=GOLD).move_to([2.9,-1.15,0])
        return Group(plot),[lambda:ShowPassingFlash(top),lambda:ShowPassingFlash(bottom),
                             lambda:ShowPassingFlash(top.copy().set_color(GOLD))]
    if kind == "conformal_ball":
        centre=np.array([2.6,.1,0])
        circle=Circle(radius=1.65,color=BLUE,fill_color=BLUE,fill_opacity=.07).move_to(centre)
        prediction=Dot(centre,color=GOLD)
        radius=Arrow(centre,centre+[1.65,0,0],buff=0,color=GREEN)
        inside=Dot(centre+[-.8,.9,0],color=GREEN,radius=.075)
        outside=Dot(centre+[1.85,1.25,0],color=RED,radius=.075)
        labels=VGroup(formula(r"\widehat G(u)",29,GOLD).next_to(prediction,DOWN,buff=.18),
                      formula(r"Qa(u)",29,GREEN).move_to(centre+[.85,-.4,0]),
                      small_label("Covered",inside.get_center()+[.2,-.35,0],25,GREEN),
                      small_label("Outside",outside.get_center()+[.1,.45,0],25,RED),
                      small_label("One ball for the entire output field",[2.6,-2.35,0],25))
        return VGroup(circle,prediction,radius,inside,outside,labels),[lambda:Indicate(prediction,color=GOLD),
                lambda:Indicate(radius,color=GREEN),lambda:Indicate(circle,color=BLUE)]
    if kind == "calibration_ranks":
        slots=VGroup();labels=VGroup()
        for j in range(10):
            centre=np.array([-.45+j*.68,.45,0])
            slots.add(Dot(centre,color=BLUE,radius=.07))
            labels.add(formula(str(j+1),21,DIM).move_to(centre+[0,-.45,0]))
        future=Circle(radius=.19,color=GOLD,stroke_width=3).move_to(slots[5])
        threshold=DashedLine([5.25,-.4,0],[5.25,1.15,0],color=GREEN)
        heads=VGroup(small_label("Rank among all ten cases",[2.6,2.05,0],27),
                     formula(r"k=9",27,GREEN).move_to([5.25,1.52,0]),
                     small_label("Gold ring: future case",[2.6,-1.55,0],26,GOLD),
                     formula(r"\mathbb P(J\le9)=9/10",30,GOLD).move_to([2.6,-2.3,0]))
        return VGroup(slots,labels,future,threshold,heads),[
            lambda:future.animate.move_to(slots[2]),lambda:future.animate.move_to(slots[8]),
            lambda:Indicate(heads[3],color=GOLD)]
    if kind == "coverage_reference":
        data=json.loads((HERE/'assets/coverage_reference.json').read_text(encoding='utf-8'))
        xx=np.array(data['conditional_p'])*100;yy=np.array(data['conditional_density'])/100
        ymax=float(yy.max())*1.15
        ax=axes(x=(86,94,2),y=(0,ymax,ymax/2),width=5.1,height=3.9).move_to([2.6,.3,0])
        curve=VMobject(color=BLUE,stroke_width=3).set_points_as_corners([ax.c2p(x,y) for x,y in zip(xx,yy)])
        mode=100*(data['k']-1)/(data['m']-1)
        marker=Dot(ax.c2p(mode,np.interp(mode,xx,yy)),color=GOLD)
        labels=VGroup(small_label("Density of conditional coverage",[2.6,2.65,0],25),
                      small_label("Conditional coverage (%)",[2.6,-2.02,0],25),
                      formula(r"p_{\rm cal}\sim\operatorname{Beta}(901,100)",27,BLUE).move_to([2.6,-2.63,0]))
        for x in (86,88,90,92,94):labels.add(formula(str(x),20,DIM).next_to(ax.c2p(x,0),DOWN,buff=.1))
        return VGroup(ax,curve,marker,labels),[lambda:ShowPassingFlash(curve.copy().set_color(GOLD)),
                lambda:Indicate(marker,color=GOLD),lambda:Indicate(labels[2],color=BLUE)]
    if kind == "calibration_protocol":
        boxes=VGroup();arrows=VGroup()
        for i,(title,sub,color) in enumerate((('Freeze','Predictor + scale',BLUE),
                ('Calibrate','Independent sample',GOLD),('Evaluate','Future case',GREEN))):
            x=.05+i*2.52
            box=RoundedRectangle(width=2.22,height=1.5,corner_radius=.09,color=color).move_to([x,.5,0])
            label=small_label(title,[x,.82,0],28,color)
            detail=prose(sub,size=23,width=2.6,color=DIM).move_to([x,.13,0])
            boxes.add(VGroup(box,label,detail))
            if i:arrows.add(Arrow([x-1.36,.5,0],[x-1.13,.5,0],buff=0,color=DIM,stroke_width=2))
        line=Arrow([-.9,-1.6,0],[6.1,-1.6,0],buff=0,color=DIM)
        label=small_label("The chronology is part of the assumption",[2.6,-2.25,0],25)
        return VGroup(boxes,arrows,line,label),[lambda:Indicate(boxes[0],color=BLUE),
                lambda:Indicate(boxes[1],color=GOLD),lambda:Indicate(boxes[2],color=GREEN)]
    if kind == "oco_state_map":
        state = RoundedRectangle(width=2.6,height=1.2,corner_radius=.09,color=BLUE).move_to([.55,1.3,0])
        reduced = RoundedRectangle(width=2.6,height=1.2,corner_radius=.09,color=GOLD).move_to([4.45,1.3,0])
        spectrum = RoundedRectangle(width=5.6,height=1.45,corner_radius=.09,color=GREEN).move_to([2.5,-1.05,0])
        labels=VGroup(small_label("Atmospheric state",state.get_center()+[0,.2,0],26,BLUE),
                      formula(r"20\text{--}24\ \mathrm{coordinates}",24,BLUE).move_to(state.get_center()+[0,-.25,0]),
                      small_label("Reduced radiance",reduced.get_center()+[0,.2,0],26,GOLD),
                      formula(r"40\ \mathrm{coefficients}",24,GOLD).move_to(reduced.get_center()+[0,-.25,0]),
                      formula(r"R(z)=\mu_y+U(\mu_z+D_z z)",30,GREEN).move_to(spectrum.get_center()+[0,.25,0]),
                      small_label("O2: 10,592 stored spectral channels",spectrum.get_center()+[0,-.35,0],25,GREEN))
        train=Arrow(state.get_right(),reduced.get_left(),buff=.13,color=GOLD)
        rebuild=Arrow(reduced.get_bottom(),spectrum.get_top()+[1.9,0,0],buff=.13,color=GREEN)
        return VGroup(state,reduced,spectrum,labels,train,rebuild),[lambda:Indicate(state,color=BLUE),
                lambda:Indicate(spectrum,color=GREEN),lambda:Indicate(train,color=GOLD)]
    if kind == "metric_ellipses":
        ax=axes(x=(-1.5,1.5,1),y=(-1.5,1.5,1),width=4.2,height=4.2).move_to([2.55,.15,0])
        centre=ax.c2p(0,0)
        circle=Circle(1.4,color=BLUE).move_to(centre)
        ellipse=Ellipse(width=2.8,height=1.4,color=GOLD).move_to(centre)
        a=Arrow(centre,ax.c2p(1,0),buff=0,color=GREEN)
        b=Arrow(centre,ax.c2p(0,.65),buff=0,color=RED)
        labels=VGroup(formula(r"e_1",27,DIM).next_to(ax.c2p(1.5,0),RIGHT,buff=.12),
                      formula(r"e_2",27,DIM).next_to(ax.c2p(0,1.5),UP,buff=.12),
                      formula(r"\|e\|_2=1",25,BLUE).move_to([5.,1.5,0]),
                      formula(r"\|\operatorname{diag}(1,2)e\|_2=1",24,GOLD).move_to([2.6,-2.35,0]),
                      formula(r"a=(1,0)",24,GREEN).next_to(a.get_end(),DOWN,buff=.2),
                      formula(r"b=(0,0.65)",24,RED).next_to(b.get_end(),LEFT,buff=.18))
        return VGroup(ax,circle,ellipse,a,b,labels),[lambda:Indicate(circle,color=BLUE),
                lambda:Indicate(ellipse,color=GOLD),lambda:Indicate(VGroup(a,b),color=INK)]
    if kind == "coordinate_selection":
        values=np.array([[.3,.9,.7,.5],[.8,.2,.6,.4],[.5,.6,.1,.8]])
        cells=VGroup(); winners=VGroup(); rows=VGroup()
        for i in range(3):
            rows.add(small_label(f"Candidate {i+1}",[.15,1.35-i*1.05,0],23))
            for j in range(4):
                winner=i==int(np.argmin(values[:,j]))
                color=GOLD if winner else BLUE
                cell=Rectangle(width=.8,height=.8,color=color,fill_color=color,fill_opacity=.3 if winner else .05)
                cell.move_to([1.7+j*1.12,1.35-i*1.05,0])
                pair=VGroup(cell,formula(f"{values[i,j]:.1f}",26,color).move_to(cell))
                cells.add(pair)
                if winner:winners.add(cell.copy())
        labels=VGroup(small_label("Illustrative validation squared losses",[2.6,2.45,0],25),
                      small_label("Output coordinate",[3.2,-1.65,0],25),
                      small_label("Choose one winner in each column",[2.6,-2.45,0],25,GOLD))
        return VGroup(cells,rows,labels),[lambda:Indicate(winners[0],color=GOLD),
                lambda:Indicate(winners,color=GOLD),lambda:Indicate(labels[2],color=GOLD)]
    if kind == "kernel_grid":
        ax=axes(x=(-2.5,4.5,1),y=(-10.6,-2.5,1),width=5.3,height=4.6).move_to([2.7,.1,0])
        expanded=VGroup(); recorded=VGroup()
        for x in range(-2,5):
            for y in range(-10,-2):
                expanded.add(Dot(ax.c2p(x,y),radius=.032,color=BLUE))
                if x in (-1,0,1,2) and y in (-8,-6,-4):
                    recorded.add(Circle(radius=.085,color=GOLD,stroke_width=2).move_to(ax.c2p(x,y)))
        labels=VGroup(formula(r"\log_2(s/s_{\rm med})",25,DIM).move_to([2.7,-2.78,0]),
                      formula(r"\log_{10}\lambda",25,DIM).move_to([-.25,2.3,0]),
                      small_label("Recorded: 12",[1.3,2.62,0],24,GOLD),
                      small_label("Expanded: 56",[4.6,2.62,0],24,BLUE))
        for x in (-2,0,2,4):labels.add(formula(str(x),18,DIM).next_to(ax.c2p(x,-10.6),DOWN,buff=.08))
        for y in (-10,-8,-6,-4):labels.add(formula(str(y),18,DIM).next_to(ax.c2p(-2.5,y),LEFT,buff=.08))
        return VGroup(ax,expanded,recorded,labels),[lambda:Indicate(recorded,color=GOLD),
                lambda:Indicate(expanded,color=BLUE),lambda:Indicate(labels[3],color=BLUE)]
    if kind == "oco_spectrum":
        receipt = json.loads((HERE/"assets/oco_spectrum_figures.json").read_text(encoding="utf-8"))
        row = next(x for x in receipt["rows"] if x["case"] == spec["case"])
        path = HERE/"assets"/row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("OCO-2 figure does not match its receipt")
        plot = ImageMobject(str(path)).set_width(7.15).move_to([2.63, .0, 0])
        # The data, labels and normalized units live in the checked scientific plot.
        top = Rectangle(width=6.6, height=2.3, color=GOLD).move_to([2.85, 1.35, 0])
        bottom = Rectangle(width=6.6, height=1.9, color=BLUE).move_to([2.85, -1.15, 0])
        scores = row["radiance_relative_errors"]
        score = formula(rf"E_{{\rm case}}:\ {100*scores['combined']:.4f}\%\quad"
                        rf"({100*scores['kernel_flow']:.4f}\%\ \mathrm{{KF}})",
                        23, GOLD, 7).move_to([2.7, -2.9, 0])
        return Group(plot, score), [lambda: ShowPassingFlash(top), lambda: ShowPassingFlash(bottom),
                                     lambda: Indicate(score, color=GOLD)]
    if kind == "mean_mismatch":
        ax = axes(x=(0, 3.6, 1), y=(0, 2.8, 1), width=5.4, height=4.2).move_to([2.65, .1, 0])
        a, b, y = ax.c2p(.7, .5), ax.c2p(1.7, .85), ax.c2p(2.8, 2.05)
        dots = VGroup(Dot(a, color=BLUE), Dot(b, color=GREEN), Dot(y, color=INK))
        delta = Arrow(a, b, buff=.07, color=GOLD)
        old = Arrow(a, y, buff=.08, color=BLUE)
        new = Arrow(b, y, buff=.08, color=GREEN)
        labels = VGroup(formula(r"M_{\rm tr}", 28, BLUE).next_to(a, DOWN, buff=.18),
                        formula(r"m_{\rm full}(X)", 27, GREEN).next_to(b, DOWN, buff=.40),
                        formula(r"G(X)", 29, INK).next_to(y, UP, buff=.18),
                        formula(r"\Delta", 27, GOLD).move_to((a+b)/2+[0,-.48,0]),
                        small_label("One schematic training row", [2.6,-2.55,0], 24))
        return VGroup(ax, old, new, delta, dots, labels), [lambda: Indicate(dots, color=INK),
                lambda: Indicate(delta, color=GOLD), lambda: Indicate(new, color=GREEN)]
    if kind == "triangle_error":
        a, b, c = np.array([.1,-1.2,0]), np.array([3.9,-.2,0]), np.array([4.8,1.55,0])
        first = Arrow(a,b,buff=0,color=BLUE)
        second = Arrow(b,c,buff=0,color=GOLD)
        total = Arrow(a,c,buff=0,color=RED)
        labels = VGroup(formula(r"e_{\rm full}",29,BLUE).move_to([2.,-1.12,0]),
                        formula(r"-c^\top\Delta",29,GOLD).move_to([5.2,.45,0]),
                        formula(r"e_{\rm tr}",29,RED).move_to([1.7,.8,0]),
                        small_label("Output space; lengths are error norms",[2.6,-2.3,0],24))
        return VGroup(first,second,total,labels), [lambda: Indicate(first,color=BLUE),
                lambda: Indicate(second,color=GOLD),lambda: Indicate(total,color=RED)]
    if kind == "fold_centering":
        cells = VGroup()
        for i in range(4):
            color = GOLD if i == 3 else BLUE
            cell = Rectangle(width=1.3,height=1.2,color=color,fill_color=color,fill_opacity=.2).move_to([.35+1.48*i,.65,0])
            cells.add(VGroup(cell,formula("4750",24,color).move_to(cell)))
        fitting = Brace(VGroup(*cells[:3]),DOWN,color=BLUE)
        pooled = Brace(cells,UP,color=GREEN)
        labels = VGroup(formula(r"\mu_F:\ 14250",29,BLUE).next_to(fitting,DOWN,buff=.15),
                        formula(r"\mu_P:\ 19000",29,GREEN).next_to(pooled,UP,buff=.15),
                        small_label("Held out",[4.8,-.55,0],24,GOLD),
                        small_label("Target centering only; other data paths remain as stated",[2.65,-2.25,0],21))
        return VGroup(cells,fitting,pooled,labels),[lambda:Indicate(pooled,color=GREEN),
                lambda:Indicate(fitting,color=BLUE),lambda:Indicate(cells[3],color=GOLD)]
    if kind == "paired_protocol":
        parts=VGroup(); stages=[]
        for y,label,color in ((.95,"Pooled",BLUE),(-.75,"Fold-local",GOLD)):
            parts.add(small_label(label,[2.65,y+.75,0],27,color))
            row=VGroup()
            for i,name in enumerate(("Field","Refiner","Stack","Kernel")):
                x=.05+i*1.65
                box=RoundedRectangle(width=1.42,height=.7,corner_radius=.07,color=color)
                box.move_to([x,y,0]); txt=small_label(name,box.get_center(),23,color)
                row.add(VGroup(box,txt))
                if i:parts.add(Arrow([x-.99,y,0],[x-.78,y,0],buff=0,color=DIM,stroke_width=2))
            parts.add(row); stages.append(row)
        labels=VGroup(small_label("Same seed, epochs and downstream choices",[2.65,-2.2,0],24),
                      formula(r"d_s=E_{s,\rm local}-E_{s,\rm pooled}",30,GOLD).move_to([2.65,-2.78,0]))
        return VGroup(parts,labels),[lambda:Indicate(stages[0],color=BLUE),
                lambda:Indicate(stages[1],color=GOLD),lambda:Indicate(labels[1],color=GOLD)]
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
                            formula("0", 18, DIM).next_to(ax.c2p(0, 0), DOWN, buff=.12),
                            formula("1", 18, DIM).next_to(ax.c2p(1, 0), DOWN, buff=.12),
                            formula(f"{ymin:.1f}", 18, DIM).next_to(ax.c2p(0, ymin), LEFT, buff=.1),
                            formula(f"{ymax:.1f}", 18, DIM).next_to(ax.c2p(0, ymax), LEFT, buff=.1))
            image = field_image(target, "viridis", float(target.min()), float(target.max()), [4.55, .25, 0], 2.5)
            ilabel = small_label("Reference stress", [4.55, 2.0, 0], 27)
            arrow = Arrow([2.35, .3, 0], [2.95, .3, 0], color=INK, buff=0)
            units = small_label("Grid: horizontal x1, vertical x2; benchmark units", [2.6,-2.2,0], 22)
            group = Group(ax, curve, labels, image, ilabel, arrow, units)
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
        parts.append(small_label("Grid: horizontal x1, vertical x2; benchmark units", [2.6, -2.62, 0], 20))
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
    if kind == "member_paths":
        def box(tex,point,color,width=1.7):
            frame=RoundedRectangle(width=width,height=.64,corner_radius=.06,color=color,
                                   fill_color=color,fill_opacity=.1).move_to(point)
            return VGroup(frame,formula(tex,25,color,width-.14).move_to(frame))
        inputs=VGroup(*[formula(r'u',28,INK).move_to([-.3,y,0]) for y in (1.35,-.15,-2.0)])
        neural=box(r'f_\theta',[2.05,1.35,0],BLUE)
        field=box(r'\widehat G',[4.6,1.35,0],BLUE)
        kernel=box(r'\widehat G_{\rm KRR}',[1.55,-.65,0],GOLD)
        refiner=box(r'f_{\rm ref}',[4.4,-.15,0],GREEN)
        feature=box(r'\varphi_\theta',[1.6,-2.0,0],BLUE)
        head=box(r'k_\varphi',[4.4,-2.0,0],GREEN)
        links=VGroup(Arrow(inputs[0].get_right(),neural.get_left(),buff=.08,color=BLUE),
            Arrow(neural.get_right(),field.get_left(),buff=.08,color=BLUE),
            Arrow(inputs[1].get_right(),kernel.get_left(),buff=.08,color=GOLD),
            Arrow(kernel.get_right(),refiner.get_left(),buff=.08,color=GOLD),
            Arrow(inputs[2].get_right(),feature.get_left(),buff=.08,color=BLUE),
            Arrow(feature.get_right(),head.get_left(),buff=.08,color=GREEN))
        direct=VMobject(color=INK,stroke_width=2).set_points_as_corners(
            [inputs[1].get_top(),[-.3,.35,0],[3.15,.35,0],[3.15,-.15,0]])
        links.add(direct,Arrow([3.15,-.15,0],refiner.get_left(),buff=.02,color=INK))
        labels=VGroup(small_label('Direct field prediction',[2.4,2.05,0],23,BLUE),
            small_label('Input and kernel prediction',[2.5,.73,0],22,GOLD),
            small_label('Learned representation',[2.5,-1.4,0],23,GREEN))
        group=VGroup(inputs,neural,field,kernel,refiner,feature,head,links,labels)
        return group,[lambda:Indicate(neural,color=BLUE),
                      lambda:AnimationGroup(Indicate(kernel,color=GOLD),Indicate(refiner,color=GREEN)),
                      lambda:AnimationGroup(Indicate(feature,color=BLUE),Indicate(head,color=GREEN))]
    if kind == "kernel_system":
        def matrix(rows,columns,point,color):
            cells=VGroup(*[Square(side_length=.37,color=color,stroke_width=1,
                fill_color=color,fill_opacity=.14+.08*((i+j)%3)).move_to([j*.39,-i*.39,0])
                for i in range(rows) for j in range(columns)])
            cells.move_to(point)
            bracket=SurroundingRectangle(cells,buff=.09,color=DIM,stroke_width=1)
            return VGroup(cells,bracket)
        A=matrix(3,3,[.65,.8,0],BLUE);alpha=matrix(3,2,[2.7,.8,0],GREEN)
        R=matrix(3,2,[4.95,.8,0],GOLD)
        labels=VGroup(formula(r'K+n\lambda I',26,BLUE).next_to(A,UP,buff=.2),
            formula(r'\alpha',29,GREEN).next_to(alpha,UP,buff=.2),
            formula(r'R',29,GOLD).next_to(R,UP,buff=.2),
            formula(r'=',32,INK).move_to([3.85,.8,0]),
            small_label('One factorization, all output columns',[2.65,2.35,0],23))
        for obj,shape in ((A,r'3\times3'),(alpha,r'3\times2'),(R,r'3\times2')):
            labels.add(formula(shape,22,DIM).next_to(obj,DOWN,buff=.15))
        query=matrix(1,3,[.65,-1.55,0],BLUE);residuals=matrix(3,2,[2.7,-1.55,0],GOLD)
        answer=matrix(1,2,[4.95,-1.55,0],GREEN)
        labels.add(formula(r'c(u)^\top',25,BLUE).next_to(query,UP,buff=.18),
            formula(r'R',26,GOLD).next_to(residuals,UP,buff=.18),
            formula(r'=',32,INK).move_to([3.85,-1.55,0]),
            formula(r'\widehat r(u)',25,GREEN).next_to(answer,UP,buff=.18),
            small_label('The same query weights combine every coordinate',[2.65,-2.62,0],21))
        group=VGroup(A,alpha,R,query,residuals,answer,labels)
        return group,[lambda:Indicate(A,color=BLUE),
            lambda:AnimationGroup(Indicate(alpha,color=GREEN),Indicate(R,color=GOLD)),
            lambda:AnimationGroup(Indicate(query,color=BLUE),Indicate(answer,color=GREEN))]
    if kind == "learned_distance":
        panels=VGroup();segments=[]
        for centre,points,title in (([.65,.45,0],[[.1,.2],[.6,.2],[.1,.7]],'Input coordinates'),
                                    ([4.45,.45,0],[[.2,.1],[1.2,.1],[.2,.35]],'Feature coordinates')):
            ax=axes(x=(0,1.5,.5),y=(0,1.5,.5),width=2.65,height=2.65).move_to(centre)
            dots=VGroup(*[Dot(ax.c2p(*p),color=color,radius=.06)
                for p,color in zip(points,(INK,BLUE,GOLD))])
            lines=VGroup(Line(dots[0].get_center(),dots[1].get_center(),color=BLUE,stroke_width=3),
                         Line(dots[0].get_center(),dots[2].get_center(),color=GOLD,stroke_width=3))
            labels=VGroup(small_label(title,[centre[0],2.15,0],22))
            for dot,letter,direction in zip(dots,('A','B','C'),(DOWN,RIGHT,UP)):
                label=formula(letter,24,INK).next_to(dot,direction,buff=.09)
                if letter=='A':
                    label.set_y(ax.get_bottom()[1]-.20)
                labels.add(label)
            panels.add(VGroup(ax,lines,dots,labels));segments.append(lines)
        transform=formula(r'\varphi(x_1,x_2)=(2x_1,x_2/2)',27,GREEN,6.7).move_to([2.6,-1.5,0])
        distances=VGroup(formula(r'AB=AC=0.5',24,INK).move_to([.65,-2.2,0]),
                         formula(r'AB=1,\ AC=0.25',24,INK).move_to([4.45,-2.2,0]))
        link=Arrow([2.1,.6,0],[2.95,.6,0],buff=.04,color=GREEN)
        group=VGroup(panels,transform,distances,link)
        return group,[lambda:Indicate(segments[0]),lambda:Indicate(segments[1],color=GREEN),
                      lambda:Indicate(transform,color=GREEN)]
    if kind == "stacking_classes":
        ax=axes(x=(-.2,2.5,.5),y=(-.2,2.5,.5),width=4.5,height=4.5).move_to([2.6,.2,0])
        points=[ax.c2p(*p) for p in ((0,0),(2,0),(0,2))]
        hull=Polygon(*points,color=BLUE,fill_color=BLUE,fill_opacity=.1)
        dots=VGroup(*[Dot(p,color=BLUE,radius=.065) for p in points])
        global_mean=Dot(ax.c2p(1,.6),color=GREEN,radius=.085)
        labels=VGroup(formula(r'f_1=(0,0)',23,BLUE).next_to(dots[0],DOWN,buff=.32),
            formula(r'f_2=(2,0)',23,BLUE).next_to(dots[1],DOWN,buff=.12),
            formula(r'f_3=(0,2)',23,BLUE).next_to(dots[2],UP,buff=.12).shift(.72*RIGHT),
            formula(r'm=(1,0.6)',24,GREEN).next_to(global_mean,DOWN,buff=.16))
        base=VGroup(ax,hull,dots,global_mean,labels)
        coordinate=Dot(ax.c2p(2,2),color=GOLD,radius=.085)
        outside=VGroup(coordinate,DashedLine(points[1],coordinate.get_center(),color=GOLD),
            DashedLine(points[2],coordinate.get_center(),color=GOLD),
            formula(r'm_{\rm coord}=(2,2)',23,GOLD).move_to([4.5,2.3,0]))
        cases=VGroup(*[Square(.3,color=BLUE,fill_color=BLUE,fill_opacity=.3).move_to([.7+j*.43,1.1,0])
                      for j in range(10)])
        selection=VGroup(cases,small_label('The same validation cases',[2.6,2.05,0],25),
            small_label('Fit weights',[.8,-.65,0],27,BLUE),
            small_label('Choose a variant',[4.3,-.65,0],27,GOLD),
            Arrow([1.5,.85,0],[.8,-.25,0],color=BLUE,buff=.05),
            Arrow([3.3,.85,0],[4.3,-.25,0],color=GOLD,buff=.05),
            small_label('These decisions share their observations',[2.6,-2.0,0],23))
        return base,[lambda:Indicate(hull,color=BLUE),lambda:FadeIn(outside),
            lambda:AnimationGroup(FadeOut(base),FadeOut(outside),FadeIn(selection))]
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
        if spec.get('show_calibration'):
            ball, _ = build_visual({'kind':'conformal_ball'})
            return group, [lambda: Indicate(mean,color=BLUE),
                lambda: Indicate(residual,color=GOLD),
                lambda: AnimationGroup(FadeOut(group),FadeIn(ball))]
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
    if kind in ("pool_matrix", "pool_seed_curves", "pool_bounds"):
        pool=json.loads((HERE/"assets/pool_geometry.json").read_text(encoding="utf-8"))
        if (pool['n_cal'],pool['n_ev']) != (1000,19000):
            raise ValueError("Unexpected empirical pool split")
        names=["MLP","MSE","Refiner","FNO","UNet","KRR"]
        colors=[BLUE,GOLD,GREEN,RED,"#C9A6E4","#D7D0B5"]
        if kind == "pool_matrix":
            values=np.asarray(pool['second_moments'])*1e4
            low,high=float(values.min()),float(values.max())
            palette=np.asarray(json.loads((HERE/'assets/colormaps.json').read_text(encoding='utf-8'))['viridis'],dtype=np.uint8)
            rgba=palette[np.rint((values-low)/(high-low)*255).astype(int)]
            picture=ImageMobject(rgba).set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest'])
            picture.set_height(3.6).move_to([2.8,.25,0])
            boundaries=VGroup();labels=VGroup()
            for a,name in enumerate(names):
                x=1.+(a+.5)*.6;y=2.05-(a+.5)*.6
                boundaries.add(Square(.6,color=INK,stroke_width=1.2).move_to([x,y,0]))
                labels.add(small_label(name,[x,-1.88,0],17))
                labels.add(small_label(name,[.5,y,0],20))
            ramp=ImageMobject(palette[::-1,None,:]).stretch_to_fit_height(3.6).stretch_to_fit_width(.16).move_to([4.97,.25,0])
            labels.add(formula(f'{high:.2f}',20,DIM).next_to(ramp,UP,buff=.1))
            labels.add(formula(f'{low:.2f}',20,DIM).next_to(ramp,DOWN,buff=.1))
            labels.add(formula(r'10^4\widehat S_{ab}',28,INK).move_to([2.8,2.55,0]))
            labels.add(small_label('Every outlined block contains ten seeds',[2.6,-2.6,0],22))
            group=Group(picture,boundaries,ramp,labels)
            return group,[lambda:Indicate(boundaries[0]),lambda:Indicate(boundaries[3]),lambda:Indicate(boundaries[5])]
        if kind == 'pool_seed_curves':
            ax=axes(x=(1,10,1),y=(4.85,5.6,.2),width=5.3,height=3.3).move_to([2.65,.6,0])
            curves=VGroup();labels=VGroup()
            for a,(configuration,name,color) in enumerate(zip(pool['configurations'],names,colors)):
                points=[ax.c2p(row['k'],100*row['mean_rms']) for row in pool['curves'][configuration]]
                curve=VMobject(color=color,stroke_width=3).set_points_as_corners(points)
                dots=VGroup(*(Dot(point,radius=.04,color=color) for point in points))
                curves.add(VGroup(curve,dots))
                labels.add(small_label(name,[.7+(a%3)*2.,-2.35-(a//3)*.42,0],21,color))
            for x in (1,2,3,5,10):
                labels.add(formula(str(x),19,DIM).next_to(ax.c2p(x,4.85),DOWN,buff=.1))
            for y in (4.9,5.1,5.3,5.5):
                labels.add(formula(f'{y:.1f}',19,DIM).next_to(ax.c2p(1,y),LEFT,buff=.1))
            labels.add(small_label('RMS relative error (%)',[2.6,2.65,0],25))
            labels.add(small_label('Seeds in an equal-weight subset',[2.6,-1.75,0],24))
            group=VGroup(ax,curves,labels)
            return group,[lambda:Indicate(curves[0]),lambda:Indicate(curves[3]),lambda:Indicate(curves[5])]
        ax=axes(x=(4.78,5.02,.05),y=(0,1.1,.5),width=5.3,height=2.6).move_to([2.6,.3,0])
        ax.y_axis.set_opacity(0)
        lower=100*pool['block']['rms_lower_bound'];upper=100*pool['optimum']['rms_upper']
        equal=100*pool['equal_rms']
        markers=VGroup();labels=VGroup()
        for value,height,label,color in ((lower,.9,'Lower bound',BLUE),(upper,.55,'Optimum',GOLD),(equal,.15,'Equal weights',GREEN)):
            marker=Line(ax.c2p(value,0),ax.c2p(value,height),color=color,stroke_width=3)
            point=Dot(ax.c2p(value,height),radius=.055,color=color)
            markers.add(VGroup(marker,point))
            text=VGroup(small_label(label,ORIGIN,21,color),formula(f'{value:.3f}'+r'\%',24,color)).arrange(DOWN,buff=.09)
            text.next_to(point,UP,buff=.12);labels.add(text)
        for x in (4.8,4.85,4.9,4.95,5.):
            labels.add(formula(f'{x:.2f}',18,DIM).next_to(ax.c2p(x,0),DOWN,buff=.12))
        labels.add(small_label('RMS relative error (%); axis shown from 4.78',[2.6,-1.8,0],23))
        labels.add(formula(rf'\text{{Gap}}={upper-lower:.3f}\text{{ percentage points}}',27,INK,6.3).move_to([2.6,-2.5,0]))
        group=VGroup(ax,markers,labels)
        return group,[lambda:Indicate(markers[0]),lambda:Indicate(markers[1]),lambda:Indicate(markers[2])]
    if kind in ("kernel_sections", "residual_function"):
        ax = axes(x=(-3, 3, 1), y=(-1.3, 1.5, 1), width=5.4, height=3.8).move_to([2.6,.15,0])
        curves = VGroup(); labels = VGroup()
        if kind == "kernel_sections":
            for center, color in zip((-1.5,0,1.5),(BLUE,GOLD,GREEN)):
                curve = ax.plot(lambda x,c=center: np.exp(-.5*(x-c)**2), x_range=[-3,3], color=color, stroke_width=3)
                curves.add(curve)
            labels.add(formula(r"k(x,z)=\exp\big(-(x-z)^2/2\big)",28,INK,6.3).move_to([2.6,2.45,0]))
            labels.add(small_label("Input coordinate z", [2.6,-2.3,0],26))
        else:
            target = lambda x: .7*np.sin(x)+.18*np.cos(3*x)
            mean = lambda x: .65*np.sin(x)
            for fn,color in ((target,INK),(mean,BLUE),(lambda x:target(x)-mean(x),GOLD)):
                curves.add(ax.plot(fn,x_range=[-3,3],color=color,stroke_width=3))
            for label,x,color in ((r"G",.2,INK),(r"m",2.6,BLUE),(r"r=G-m",4.8,GOLD)):
                labels.add(formula(label,30,color).move_to([x,2.35,0]))
            labels.add(small_label("Toy functions of a scalar input", [2.6,-2.3,0],24))
        for x in (-3,0,3):
            labels.add(formula(str(x),18,DIM).next_to(ax.c2p(x,0),DOWN,buff=.12))
        group = VGroup(ax,curves,labels)
        return group,[lambda:Indicate(curves[0]),lambda:Indicate(curves[1]),lambda:Indicate(curves[2])]
    if kind == "feature_circle":
        center=np.array([2.5,.05,0]);radius=1.75
        circle=Circle(radius,color=DIM).move_to(center)
        angle=np.pi/4
        query_end=center+radius*np.array([np.cos(angle),np.sin(angle),0])
        horizontal=Arrow(center,center+radius*RIGHT,buff=0,color=BLUE,stroke_width=4)
        vertical=Arrow(center,center+radius*UP,buff=0,color=GREEN,stroke_width=4)
        query=Arrow(center,query_end,buff=0,color=GOLD,stroke_width=4)
        horizontal_part=DashedLine(query_end,center+[radius*np.cos(angle),0,0],color=DIM)
        labels=VGroup(formula(r"\varphi(0)",26,BLUE).next_to(horizontal.get_end(),RIGHT,buff=.15),
                      formula(r"\varphi(\pi/2)",26,GREEN).next_to(vertical.get_end(),UP,buff=.15),
                      formula(r"\varphi(u)",27,GOLD).next_to(query_end,RIGHT,buff=.12),
                      formula(r"\varphi(u)=(\cos u,\sin u)",29,INK,6.2).move_to([2.6,-2.35,0]))
        group=VGroup(circle,horizontal,vertical,query,horizontal_part,labels)
        return group,[lambda:Indicate(query,color=GOLD),lambda:Indicate(horizontal_part),
                      lambda:AnimationGroup(Indicate(horizontal),Indicate(vertical))]
    if kind in ("ensemble_segment", "ensemble_hull"):
        from ensemble_geometry import two_member
        first = np.array([1., 0.])
        second = np.array([1.2, .4]) if spec.get("variant") == "aligned" else np.array([0., 2.])
        result = two_member(first, second)
        ax = axes(x=(-.4, 2.6, 1), y=(-.4, 2.6, 1), width=3.8, height=3.8).move_to([2.6, .05, 0])
        origin = ax.c2p(0, 0)
        v1 = Arrow(origin, ax.c2p(*first), buff=0, color=BLUE, stroke_width=4)
        v2 = Arrow(origin, ax.c2p(*second), buff=0, color=GOLD, stroke_width=4)
        segment = Line(ax.c2p(*first), ax.c2p(*second), color=GREEN, stroke_width=3)
        optimum = ax.c2p(*result['point'])
        error = DashedLine(origin, optimum, color=RED, stroke_width=3)
        dot = Dot(ax.c2p(*first), color=RED, radius=.075)
        labels = VGroup(formula(r"\rho_1", 28, BLUE).next_to(v1.get_end(), DOWN, buff=.18),
                        formula(r"\rho_2", 28, GOLD).next_to(v2.get_end(), UP, buff=.17).shift(.32*LEFT),
                        formula(rf"t^\star={result['weight']:.3f}", 28, GREEN).move_to([4.8, -.4, 0]),
                        formula(rf"\min\|\rho_w\|^2={result['squared_error']:.3f}", 28, RED).move_to([2.6, -2.5, 0]))
        group = VGroup(ax, segment, v1, v2, error, dot, labels)
        if spec.get("intro"):
            group.remove(error, dot)
            labels[2].set_opacity(0); labels[3].set_opacity(0)
            return group, [lambda: Indicate(v1, color=BLUE), lambda: Indicate(v2, color=GOLD),
                           lambda: Indicate(segment, color=GREEN)]
        if spec.get("variant") == "aligned":
            tangent_end = ax.c2p(1, 1.2)
            tangent_vector = Arrow(origin, tangent_end, buff=0, color=GOLD, stroke_width=4)
            tangent_segment = Line(ax.c2p(*first), tangent_end, color=GREEN, stroke_width=3)
            zero_segment = Line(origin, tangent_end, color=GREEN, stroke_width=3)
            zero_error = formula(r"\min\|\rho_w\|^2=0", 28, RED).move_to(labels[3])
            zero_member = formula(r"\rho_1=0", 28, BLUE).next_to(origin, DOWN, buff=.2).shift(.55*RIGHT)
            return group, [lambda: Indicate(v2, color=GOLD),
                lambda: AnimationGroup(Transform(v2, tangent_vector), Transform(segment, tangent_segment),
                    labels[1].animate.next_to(tangent_end, UP, buff=.17)),
                lambda: AnimationGroup(FadeOut(v1), FadeOut(error), dot.animate.move_to(origin),
                    Transform(segment, zero_segment), Transform(labels[0], zero_member),
                    Transform(labels[3], zero_error))]
        if kind == "ensemble_hull":
            third = ax.c2p(2, 1)
            hull = Polygon(ax.c2p(*first), ax.c2p(*second), third, color=GREEN,
                           fill_color=GREEN, fill_opacity=.12, stroke_width=2)
            extra = Arrow(origin, third, buff=0, color=DIM, stroke_width=2)
            group.add_to_back(hull); group.add(extra)
        path = Line(ax.c2p(*first), optimum)
        return group, [lambda: Indicate(v1, color=BLUE), lambda: Indicate(v2, color=GOLD),
                       lambda: MoveAlongPath(dot, path) if path.get_length() > 1e-9 else Indicate(dot)]
    if kind == "second_moment_toy":
        matrix = np.array([[1., 0.], [0., 4.]])
        cells = VGroup(); labels = VGroup()
        for i in range(2):
            for j in range(2):
                center = [1.65+j*1.7, 1.0-i*1.7, 0]
                cell = Square(1.65, color=DIM, fill_color=BLUE if i == j else GOLD,
                              fill_opacity=.13+.08*matrix[i,j]).move_to(center)
                cells.add(cell)
                labels.add(formula(f"{matrix[i,j]:g}", 44, INK).move_to(center))
        labels.add(formula(r"S=\begin{pmatrix}1&0\\0&4\end{pmatrix}", 31, INK).move_to([2.5, -2.2, 0]))
        labels.add(small_label("Squared lengths on the diagonal", [2.5, 2.35, 0], 25))
        group = VGroup(cells, labels)
        return group, [lambda: Indicate(cells[0], color=BLUE), lambda: Indicate(cells[3], color=BLUE),
                       lambda: AnimationGroup(Indicate(cells[1], color=GOLD), Indicate(cells[2], color=GOLD))]
    if kind == "equicorrelation_curve":
        correlation = .6
        ax = axes(x=(1, 30, 10), y=(0, 1.1, .2), width=5.3, height=3.7).move_to([2.6,.15,0])
        curve = ax.plot(lambda m: correlation+(1-correlation)/m, x_range=[1,30], color=GOLD, stroke_width=3)
        floor = DashedLine(ax.c2p(1,.6), ax.c2p(30,.6), color=BLUE)
        labels = VGroup(small_label("Member count", [2.6,-2.25,0], 26),
                        small_label("Normalized squared error", [2.6,2.45,0], 26),
                        formula(r"\varrho=0.6", 25, BLUE).next_to(floor, UP, buff=.15))
        for x in (1,10,20,30):
            labels.add(formula(str(x), 19, DIM).next_to(ax.c2p(x,0), DOWN, buff=.1))
        for y in (0,.2,.4,.6,.8,1):
            labels.add(formula(f"{y:g}", 19, DIM).next_to(ax.c2p(1,y), LEFT, buff=.1))
        dot = Dot(ax.c2p(1,1), radius=.07, color=INK)
        group = VGroup(ax, curve, floor, labels, dot)
        return group, [lambda: Indicate(dot), lambda: MoveAlongPath(dot, curve), lambda: Indicate(floor)]
    if kind == "error_metric_geometry":
        ax = axes(x=(.5, 2.5, 1), y=(0, 2.3, 1), width=3.8, height=3.4).move_to([2.4,.2,0])
        zero = Dot(ax.c2p(1,0), radius=.07, color=BLUE)
        bar = Rectangle(width=.8, height=ax.c2p(2,2)[1]-ax.c2p(2,0)[1],
                        color=GOLD, fill_color=GOLD, fill_opacity=.5)
        bar.move_to((ax.c2p(2,2)+ax.c2p(2,0))/2)
        mean = DashedLine(ax.c2p(.5,1), ax.c2p(2.5,1), color=GREEN)
        rms = DashedLine(ax.c2p(.5,np.sqrt(2)), ax.c2p(2.5,np.sqrt(2)), color=RED)
        labels = VGroup(small_label("Case 1", ax.c2p(1,-.25), 25),
                        small_label("Case 2", ax.c2p(2,-.25), 25),
                        formula(r"\mathcal E_1=1", 27, GREEN).next_to(mean, RIGHT, buff=.15),
                        formula(r"\mathcal E_2=\sqrt2", 27, RED).next_to(rms, RIGHT, buff=.15),
                        small_label("Two equally likely relative errors", [2.6,2.35,0], 25))
        group = VGroup(ax, zero, bar, mean, rms, labels)
        return group, [lambda: Indicate(bar), lambda: Indicate(rms, color=RED), lambda: Indicate(mean, color=GREEN)]
    if kind == "kernel_projection":
        from kernel_geometry import geometry
        ridge = spec.get("ridge", 0.)
        repeated = spec.get("repeated", False)
        result = geometry([[1, 0], [1, 0]] if repeated else [[1, 0]], [3, 2], ridge)
        coefficient = sum(result["coefficients"])
        ax = axes(x=(-.3, 4, 1), y=(-.3, 3, 1), width=5.16, height=3.96).move_to([2.7, .15, 0])
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
        if repeated:
            labels[1].become(formula(r"\phi(x_1)=\phi(x_2)=(1,0)", 22, BLUE, 6.2).move_to([2.6,-1.78,0]))
            labels[2].become(formula(rf"\sum_i c_i\phi(x_i)=({coefficient:g},0)", 22, GREEN, 6.2).move_to([2.6,-2.20,0]))
        group = VGroup(ax, projection, train, fitted, query, gap, labels)
        if spec.get("focus") == "coefficients":
            gap.set_opacity(0); labels[3].set_opacity(0); labels[4].set_opacity(0)
        if spec.get("focus") == "norm":
            horizontal_gap = Arrow(ax.c2p(coefficient,0),ax.c2p(3,0),buff=0,color=RED,stroke_width=3)
            vertical_gap = Arrow(ax.c2p(3,0),ax.c2p(3,2),buff=0,color=GREEN,stroke_width=3)
            group.add(horizontal_gap,vertical_gap)
        if spec.get("focus") == "coefficients":
            projected = Arrow(origin,ax.c2p(3,0),buff=0,color=GREEN,stroke_width=4)
            zero_label = formula(r'\lambda=0:\ c=3',23,GREEN).move_to(labels[2])
            return group,[lambda:Indicate(query,color=GOLD),lambda:Indicate(fitted,color=GREEN),
                          lambda:AnimationGroup(Transform(fitted,projected),Transform(labels[2],zero_label),Indicate(projection))]
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
                       lambda: AnimationGroup(Transform(r, target), FadeOut(projection),
                           labels[2].animate.next_to(target.get_end(), RIGHT, buff=.1))]
    if kind == "invisible_residuals":
        ax = axes(x=(-1, 3.5, 1), y=(-2, 2.5, 1), width=4.1, height=4.1).move_to([2.6, .05, 0])
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
                           labels[2].animate.next_to(mid, DOWN, buff=.25),
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
        equation = formula(segment["math"], size=35, max_width=4.8, shrink=False).move_to([-3.85, -.65, 0])
        clearance = line.get_bottom()[1]-.35-equation.get_top()[1]
        if clearance < 0:
            equation.shift(UP*clearance)
        if equation.get_bottom()[1] < -2.7:
            raise ValueError('Reflow the tall formula or its heading: '+segment['math'])
        return VGroup(line, equation)


class BoardPreview(BoardMixin, Scene):
    def construct(self):
        key = os.environ.get("NMKC_BOARD", DATA["boards"][0]["key"])
        board = next(b for b in DATA["boards"] if b["key"] == key)
        self.frame(board)
        index = int(os.environ.get("NMKC_SEGMENT", "0"))
        self.add(self.segment_text(board["segments"][index]))


class NotationProbe(Scene):
    def construct(self):
        self.add(prose('LaTeX notation: native-resolution conversion check', size=36).to_edge(UP))
        expressions = [r'\frac{a+b}{c+d}\qquad\tfrac{1}{2}\qquad\dfrac{x^2}{1+x}',
                       r'\sqrt{x^2+y^2}\qquad\overline{AB}\qquad\underline{a+b}',
                       r'\binom{n}{k}\qquad\begin{pmatrix}a&b\\c&d\end{pmatrix}',
                       r'\begin{array}{c|c}a&b\\\hline c&d\end{array}\qquad x_i^{2}+x_{i+1}^{-1}']
        for expression, y in zip(expressions, (2.0,.6,-.9,-2.4)):
            self.add(formula(expression,size=42,max_width=12).move_to([0,y,0]))


class BoardSamples(BoardMixin, Scene):
    def construct(self):
        out = HERE / "media" / "board_samples" / CHAPTER
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        selected_key = os.environ.get("NMKC_BOARD")
        boards = [b for b in DATA["boards"] if selected_key is None or b["key"] == selected_key]
        if not boards:
            raise ValueError(f"Unknown board: {selected_key}")
        for board in boards:
            self.clear()
            _, effects = self.frame(board)
            text = None
            for i, segment in enumerate(board["segments"], 1):
                if text is not None:
                    self.remove(text)
                text = self.segment_text(segment)
                self.add(text)
                # -s skips encoding but applies the actual permanent changes
                # from each animation. Later samples retain earlier geometry.
                self.play(effects[min(i-1,len(effects)-1)](), run_time=1.4)
                self.renderer.update_frame(self)
                path = out / f'{board["key"]}_{i:02}.png'
                self.renderer.camera.get_image().save(path)
                rows.append({"board": board["key"], "segment": i, "path": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (out / "manifest.json").write_text(json.dumps({
            "kind": "author_layout_samples", "chapter": CHAPTER,
            "selected_board": selected_key,
            "human_visual_review": "PENDING", "rows": rows}, indent=2), encoding="utf-8")


class LectureChapter(BoardMixin, Scene):
    def construct(self):
        selected_key = os.environ.get("NMKC_BOARD")
        boards = [b for b in DATA["boards"] if selected_key is None or b["key"] == selected_key]
        if not boards:
            raise ValueError(f"Unknown board: {selected_key}")
        timing = []
        samples = HERE / "media" / "board_samples" / CHAPTER
        samples.mkdir(parents=True, exist_ok=True)
        sample_rows = []
        # Fail before spending render time if any recording is stale or missing.
        durations = {}
        for board in boards:
            for i, segment in enumerate(board["segments"], 1):
                key = f'{board["key"]}_{i:02}'
                durations[key] = validate(HERE / "audio" / CHAPTER / (key + ".wav"), segment["say"])
        for board in boards:
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
                self.renderer.update_frame(self)
                sample_path = samples / (key + ".png")
                self.renderer.camera.get_image().save(sample_path)
                sample_rows.append({"key": key, "path": sample_path.name,
                                    "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                                    "state": "after_segment_animation_and_hold"})
                timing.append({"key": key, "start": start, "voice_start": voice_start,
                               "voice_seconds": duration, "end": self.renderer.time,
                               "board": board["key"]})
        out = HERE / "timing"
        out.mkdir(exist_ok=True)
        (out / f"chapter{CHAPTER}.json").write_text(json.dumps({
            "chapter": CHAPTER, "selected_board": selected_key, "script_sha256": hashlib.sha256(
                (HERE/"chapters"/(CHAPTER+".json")).read_bytes()).hexdigest(),
            "segments": timing, "seconds": self.renderer.time}, indent=2), encoding="utf-8")
        (samples / "manifest.json").write_text(json.dumps({
            "kind": "frames_from_narrated_render", "chapter": CHAPTER,
            "selected_board": selected_key,
            "human_visual_review": "PENDING", "rows": sample_rows}, indent=2), encoding="utf-8")
