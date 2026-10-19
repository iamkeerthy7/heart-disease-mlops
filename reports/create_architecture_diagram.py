from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).resolve().parent / "figures" / "architecture_diagram.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1600, 820
im = Image.new("RGB", (W, H), "#f7f9fc")
d = ImageDraw.Draw(im)

def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for f in candidates:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()

title_font = font(36, True)
head_font = font(22, True)
body_font = font(18)

d.text((W // 2, 30), "Heart Disease MLOps — End-to-End Architecture",
       anchor="ma", fill="#17324d", font=title_font)

boxes = [
    (35, 145, 345, 335, "DATA & EDA",
     ["UCI Cleveland dataset", "Cleaning and exploration", "EDA plots"]),
    (445, 145, 755, 335, "MODEL TRAINING",
     ["Preprocessing pipeline", "Logistic Regression + RF", "CV and hyperparameter tuning"]),
    (855, 145, 1165, 335, "MLFLOW",
     ["Parameters and metrics", "Plots and run artifacts", "Saved model pipeline"]),
    (1265, 145, 1565, 335, "FASTAPI",
     ["JSON /predict endpoint", "Prediction + confidence", "Request logging"]),
    (245, 485, 655, 675, "CI/CD + DOCKER",
     ["GitHub Actions: lint and tests", "Train model and build image"]),
    (945, 485, 1355, 675, "KUBERNETES + MONITORING",
     ["Deployment, Service and Ingress", "/health and /metrics", "Monitoring dashboard"]),
]
fills = ["#e7f0ff", "#e6f6ed", "#fff1d6", "#efe8ff", "#e7f0ff", "#e6f6ed"]

for (x1, y1, x2, y2, heading, lines), fill in zip(boxes, fills):
    d.rounded_rectangle((x1, y1, x2, y2), radius=20, fill=fill,
                        outline="#58718a", width=3)
    d.text(((x1+x2)//2, y1+25), heading, anchor="ma",
           fill="#17324d", font=head_font)
    y = y1 + 72
    for line in lines:
        d.text(((x1+x2)//2, y), line, anchor="ma",
               fill="#263849", font=body_font)
        y += 31

def arrow(start, end):
    d.line([start, end], fill="#58718a", width=5)
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    size = 17
    p1 = (end[0]-size*math.cos(angle-math.pi/6),
          end[1]-size*math.sin(angle-math.pi/6))
    p2 = (end[0]-size*math.cos(angle+math.pi/6),
          end[1]-size*math.sin(angle+math.pi/6))
    d.polygon([end, p1, p2], fill="#58718a")

arrow((345, 240), (445, 240))
arrow((755, 240), (855, 240))
arrow((1165, 240), (1265, 240))
arrow((450, 335), (450, 485))
arrow((655, 580), (945, 580))
arrow((1415, 335), (1415, 580))

d.text((W//2, 755),
       "Educational demonstration only — not a clinically validated diagnostic system",
       anchor="ma", fill="#52677f", font=body_font)

im.save(OUT, optimize=True)
print(f"Created: {OUT}")
