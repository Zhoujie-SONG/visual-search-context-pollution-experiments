import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from context_pollution_utils import assert_can_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy vs number of irrelevant crops.")
    parser.add_argument("--summary_csv", default="outputs/summary_context_pollution.csv")
    parser.add_argument("--output_png", default="outputs/accuracy_vs_irrelevant_crops.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_png = Path(args.output_png)
    assert_can_write(output_png, args.overwrite)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.summary_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    original_acc = None
    gt_acc = None
    points = []
    for row in rows:
        condition = row["condition"]
        acc = float(row["accuracy"])
        if condition == "original_only":
            original_acc = acc
        elif condition == "gt_crop_only":
            gt_acc = acc
            points.append((0, acc, condition))
        elif condition.startswith("gt_plus_"):
            points.append((int(row["k_irrelevant"]), acc, condition))

    points = sorted(points)
    width, height = 1100, 760
    margin_left, margin_right, margin_top, margin_bottom = 100, 60, 80, 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 28), "Accuracy vs irrelevant evidence crops", fill=(0, 0, 0))

    max_x = max([x for x, _, _ in points] + [8])
    min_y, max_y = 0.0, 1.0

    def xy(x_value: float, y_value: float):
        x = margin_left + (x_value / max_x) * plot_w if max_x else margin_left
        y = margin_top + (max_y - y_value) / (max_y - min_y) * plot_h
        return int(round(x)), int(round(y))

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(0, 0, 0), width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(0, 0, 0), width=2)
    for tick in range(0, max_x + 1):
        x, y0 = xy(tick, 0)
        draw.line((x, y0, x, y0 + 6), fill=(0, 0, 0))
        draw.text((x - 4, y0 + 12), str(tick), fill=(0, 0, 0))
    for tick in range(0, 11):
        value = tick / 10
        x0, y = xy(0, value)
        draw.line((x0 - 6, y, x0, y), fill=(0, 0, 0))
        draw.text((30, y - 7), f"{value:.1f}", fill=(0, 0, 0))
    draw.text((margin_left + plot_w // 2 - 80, height - 45), "Number of irrelevant crops", fill=(0, 0, 0))
    draw.text((16, margin_top + plot_h // 2), "Accuracy", fill=(0, 0, 0))

    if original_acc is not None:
        _, y = xy(0, original_acc)
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(150, 150, 150), width=2)
        draw.text((margin_left + plot_w - 210, y - 18), f"original only: {original_acc:.3f}", fill=(80, 80, 80))
    if gt_acc is not None:
        _, y = xy(0, gt_acc)
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(60, 120, 200), width=2)
        draw.text((margin_left + plot_w - 210, y + 4), f"GT crop only: {gt_acc:.3f}", fill=(30, 80, 160))

    if points:
        coords = [xy(x, y) for x, y, _ in points]
        if len(coords) > 1:
            draw.line(coords, fill=(200, 50, 50), width=4)
        for (x_value, y_value, _), (x, y) in zip(points, coords):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(200, 50, 50), outline=(120, 0, 0))
            draw.text((x + 10, y - 18), f"{y_value:.3f}", fill=(120, 0, 0))

    image.save(output_png)
    print(f"wrote plot: {output_png}")


if __name__ == "__main__":
    main()
