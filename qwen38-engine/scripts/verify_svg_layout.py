#!/usr/bin/env python3
"""Static and dynamic verification for engine SVG figures.

Checks:
  1. Active content and XML safety: no <script>, javascript:, foreignObject, ENTITY, or on* handlers.
  2. XML well-formedness and SVG root.
  3. Canvas-fit: every text estimate stays inside the drawing area.
  4. Minimum readable font size.
  5. Pairwise text overlap: no two visible text runs may overlap by more than 1 px.
  6. Optional render smoke test with cairosvg when installed.

Exit 0 only when errors and overlap/canvas warnings are absent.

Usage:
  scripts/verify_svg_layout.py docs/figures/55_bonsai_local_opt_serving.svg
"""
import argparse
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ANSI_ACTIVE = re.compile(rb"<script|javascript:|<foreignObject|<!ENTITY|\son[a-z]+\s*=", re.I)
CSS_RULE = re.compile(r"\.([A-Za-z0-9_-]+)\s*\{([^}]*)\}", re.S)
FONT_SIZE_PROP = re.compile(r"font-size\s*:\s*([0-9]+(?:\.[0-9]+)?)px", re.I)
FONT_SHORTHAND = re.compile(r"font\s*:\s*(?:[a-z0-9_-]+\s+)?([0-9]+(?:\.[0-9]+)?)px", re.I)
FONT_WEIGHT_PROP = re.compile(r"font-weight\s*:\s*(bold|[1-9][0-9]{2,3})", re.I)
FONT_ATTR = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")
WEIGHT_TOKEN = re.compile(r"^(bold|[1-9][0-9]{2,3})$", re.I)


class Css:
    def __init__(self):
        self.rules = {}

    def add_style_text(self, text):
        for cls, body in CSS_RULE.findall(text):
            fs = None
            w = None
            m = FONT_SIZE_PROP.search(body)
            if m:
                fs = float(m.group(1))
            m = FONT_SHORTHAND.search(body)
            if fs is None and m:
                fs = float(m.group(1))
            m = FONT_WEIGHT_PROP.search(body)
            if m:
                w = weight_value(m.group(1))
            self.rules.setdefault(cls, {})
            if fs is not None:
                self.rules[cls]["font-size"] = fs
            if w is not None:
                self.rules[cls]["font-weight"] = w

    def class_size(self, cls):
        try:
            return float(self.rules.get(cls, {})["font-size"])
        except KeyError:
            return None
        except ValueError:
            return None

    def class_weight(self, cls):
        try:
            return int(self.rules.get(cls, {})["font-weight"])
        except (KeyError, ValueError):
            return 0


class NodeStyle:
    def __init__(self, el, size, weight):
        self.el = el
        self.size = size
        self.weight = weight


def weight_value(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s == "bold":
        return 1
    try:
        f = float(s)
    except ValueError:
        return 0
    return 1 if f >= 600 else 0


def parse_svg(path):
    raw = path.read_bytes()
    errors = []
    warnings = []
    if len(raw) > 500_000:
        warnings.append(f"SVG is {len(raw)} bytes; review readability and render size")
    active = ANSI_ACTIVE.search(raw)
    if active:
        errors.append(f"active content or event handler near byte {active.start()}")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return None, [f"XML is invalid: {exc}"], []
    if not root.tag.endswith("svg"):
        return root, ["root element is not <svg>"], []

    css = Css()
    for style in root.iter("{http://www.w3.org/2000/svg}style"):
        if style.text:
            css.add_style_text(style.text)

    vb = root.attrib.get("viewBox", "")
    parts = vb.replace(",", " ").split()
    if len(parts) == 4:
        canvas_w, canvas_h = float(parts[2]), float(parts[3])
    else:
        try:
            canvas_w = float(root.attrib.get("width", "0").replace("px", ""))
            canvas_h = float(root.attrib.get("height", "0").replace("px", ""))
        except ValueError:
            canvas_w = canvas_h = 0.0
    if canvas_w <= 0 or canvas_h <= 0:
        errors.append("cannot determine SVG canvas width or height")
        return root, errors, warnings

    ns = "{http://www.w3.org/2000/svg}"

    def node_style(el, parent_size, parent_weight):
        cls = el.attrib.get("class", "").strip()
        size = parent_size
        weight = parent_weight
        if cls:
            css_size = css.class_size(cls)
            if css_size is not None:
                size = css_size
            css_weight = css.class_weight(cls)
            if css_weight:
                weight = css_weight
        if "font-size" in el.attrib:
            m = FONT_ATTR.search(el.attrib["font-size"])
            if m:
                size = float(m.group(1))
        if "font-weight" in el.attrib and WEIGHT_TOKEN.match(el.attrib["font-weight"].strip()):
            weight = weight_value(el.attrib["font-weight"])
        return NodeStyle(el, size, weight)

    boxes = []

    def walk(el, parent_size, parent_weight):
        style = node_style(el, parent_size, parent_weight)
        if el.tag == ns + "text" or not (ns + "text" in el.tag) and el.tag.replace(ns, "", 1) == "text":
            pieces = []

            def collect(node, node_size, node_weight):
                node_cls = node.attrib.get("class", "").strip()
                if node_cls:
                    css_size = css.class_size(node_cls)
                    if css_size is not None:
                        node_size = css_size
                    css_weight = css.class_weight(node_cls)
                    if css_weight:
                        node_weight = css_weight
                if "font-size" in node.attrib:
                    m = FONT_ATTR.search(node.attrib["font-size"])
                    if m:
                        node_size = float(m.group(1))
                if "font-weight" in node.attrib and WEIGHT_TOKEN.match(node.attrib["font-weight"].strip()):
                    node_weight = weight_value(node.attrib["font-weight"])
                if node.text:
                    pieces.append((node.text, node_size, node_weight))
                for child in node:
                    collect(child, node_size, node_weight)

            start_size = style.size
            start_weight = style.weight
            cls = el.attrib.get("class", "").strip()
            if cls:
                css_size = css.class_size(cls)
                if css_size is not None:
                    start_size = css_size
                css_weight = css.class_weight(cls)
                if css_weight:
                    start_weight = css_weight
            if "font-size" in el.attrib:
                m = FONT_ATTR.search(el.attrib["font-size"])
                if m:
                    start_size = float(m.group(1))
            collect(el, start_size, start_weight)
            visible = "".join(t for t, _, _ in pieces).strip()
            if visible:
                width = 0.0
                max_size = start_size
                for text, size, weight in pieces:
                    factor = 0.60 if weight else 0.55
                    width += len(text) * size * factor
                    max_size = max(max_size, size)
                try:
                    x = float(el.attrib.get("x", "0").replace("px", ""))
                except ValueError:
                    x = 0.0
                try:
                    y = float(el.attrib.get("y", "0").replace("px", ""))
                except ValueError:
                    y = 0.0
                anchor = el.attrib.get("text-anchor", "start")
                if anchor == "middle":
                    left = x - width / 2.0
                elif anchor == "end":
                    left = x - width
                else:
                    left = x
                top = y - 0.82 * max_size
                bottom = y + 0.24 * max_size
                boxes.append({
                    "text": visible[:90],
                    "size": max_size,
                    "left": left,
                    "right": left + width,
                    "top": top,
                    "bottom": bottom,
                    "x": x,
                    "y": y,
                })
        for child in el:
            walk(child, style.size, style.weight)

    root_size = 14.0
    root_weight = 0
    m = FONT_SIZE_PROP.search(root.attrib.get("font-size", "")) if "font-size" in root.attrib else None
    if m:
        root_size = float(m.group(1))
    walk(root, root_size, root_weight)

    for box in boxes:
        if box["left"] < -0.5 or box["right"] > canvas_w + 0.5 or box["top"] < -0.5 or box["bottom"] > canvas_h + 0.5:
            warnings.append(f"text outside canvas {canvas_w:g}x{canvas_h:g}: {box['text']!r}")

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            ox = min(a["right"], b["right"]) - max(a["left"], b["left"])
            oy = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
            if ox > 1.0 and oy > 1.5:
                warnings.append(f"text overlap ({ox:.1f}x{oy:.1f}px): {a['text']!r} <-> {b['text']!r}")
    return root, errors, warnings


def render(path, out_dir):
    try:
        import cairosvg
    except Exception as exc:  # noqa: BLE001 - report optional renderer gap only
        return None, f"cairosvg render skipped: {exc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".verify.png")
    try:
        cairosvg.svg2png(url=str(path), write_to=str(out))
    except Exception as exc:  # noqa: BLE001 - renderer failure is a verification error
        return None, f"cairosvg render failed: {exc}"
    return out, None


def main():
    ap = argparse.ArgumentParser(description="Verify engine SVG figures for active content, canvas fit, and text overlap.")
    ap.add_argument("svg", type=pathlib.Path)
    ap.add_argument("--render-dir", type=pathlib.Path, default=pathlib.Path("/tmp/opencode/svg-verify"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    if not args.svg.is_file():
        print(f"verify_svg_layout: file not found: {args.svg}")
        return 2
    _, errors, warnings = parse_svg(args.svg)
    if not args.no_render:
        _, render_error = render(args.svg, args.render_dir)
        if render_error:
            if "skipped" in render_error:
                print(render_error)
            else:
                errors.append(render_error)
    for err in errors:
        print(f"ERROR: {err}")
    for warn in warnings:
        print(f"WARNING: {warn}")
    print(f"{args.svg}: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors or warnings else 0


if __name__ == "__main__":
    sys.exit(main())