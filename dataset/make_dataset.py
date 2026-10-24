"""The benchmark's dataset: 20 synthetic 1024x1024 pictures of coloured shapes, each with a caption
saying what's in it. Deterministic (seeded), no people, nothing from anyone's data."""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

out = Path(__file__).resolve().parents[1] / "work" / "data" / "bench20"
out.mkdir(parents=True, exist_ok=True)
COLOURS = {"red": (210, 50, 50), "blue": (50, 90, 210), "green": (50, 170, 80), "yellow": (235, 200, 40),
           "purple": (140, 60, 180), "orange": (240, 130, 30), "teal": (30, 160, 160), "white": (240, 240, 235)}
GROUNDS = {"a pale grey studio backdrop": (205, 205, 210), "a dark navy background": (25, 30, 60),
           "a warm beige wall": (225, 205, 175), "a soft green field": (110, 160, 90)}
for i in range(20):
    rnd = random.Random(i)
    ground, gcol = rnd.choice(list(GROUNDS.items()))
    im = Image.new("RGB", (1024, 1024), gcol)
    d = ImageDraw.Draw(im)
    parts = []
    for _ in range(rnd.randint(2, 4)):
        name, col = rnd.choice(list(COLOURS.items()))
        kind = rnd.choice(["circle", "square", "triangle"])
        x, y, s = rnd.randint(80, 700), rnd.randint(80, 700), rnd.randint(140, 320)
        if kind == "circle":
            d.ellipse((x, y, x + s, y + s), fill=col)
        elif kind == "square":
            d.rectangle((x, y, x + s, y + s), fill=col)
        else:
            d.polygon([(x + s // 2, y), (x, y + s), (x + s, y + s)], fill=col)
        parts.append(f"a {name} {kind}")
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    im.save(out / f"shapes_{i:02d}.png")
    (out / f"shapes_{i:02d}.txt").write_text(f"zbshapes, {', '.join(parts)} on {ground}, flat colours, simple composition")
print(len(list(out.glob('*.png'))), "pictures")
