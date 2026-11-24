"""Write figures/index.html: every campaign figure with its standalone caption, grouped by experiment.

    python experiments/paper_campaign/plots/gallery.py
"""
import html

from common import FIGURES

GROUPS = [("Overview", ("overview",)), ("Degree dispersion", ("degree_",)), ("Network size", ("size_",)),
          ("Composition and readouts", ("composition_",)), ("Local laws and temporal structure", ("local_", "temporal_", "lag_", "covariance_")),
          ("Recording protocol", ("record_",)), ("Other couplings and rates", ("generality_",)),
          ("Fixed-point modes", ("spectral_",)), ("Controls and bookkeeping", ("check_",))]


def main():
    names = sorted(p.stem for p in FIGURES.glob("*.png"))
    parts = ["<!doctype html><meta charset='utf-8'><title>Paper campaign figures</title>",
             "<style>body{font:15px/1.5 -apple-system,Helvetica,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 16px;color:#1f2430}"
             "img{max-width:100%;border:1px solid #d5dae3}figure{margin:0 0 2.5rem}figcaption{font-size:14px;color:#3a4252}"
             "h2{margin-top:3rem;border-bottom:1px solid #d5dae3}code{font-size:13px}</style>",
             f"<h1>Paper campaign figures</h1><p>{len(names)} figures. Each has a vector PDF and a caption file beside the PNG.</p>"]
    used = set()
    for title, prefixes in GROUPS:
        chosen = [n for n in names if n.startswith(prefixes) and n not in used]
        if not chosen:
            continue
        parts.append(f"<h2>{html.escape(title)}</h2>")
        for n in chosen:
            used.add(n)
            caption = (FIGURES / f"{n}.txt").read_text().strip() if (FIGURES / f"{n}.txt").exists() else ""
            parts.append(f"<figure><img src='{n}.png' alt='{html.escape(n)}'><figcaption><code>{n}.pdf</code><br>{html.escape(caption)}</figcaption></figure>")
    assert used == set(names), set(names) - used
    (FIGURES / "index.html").write_text("\n".join(parts) + "\n")
    print(f"wrote index.html with {len(names)} figures")


if __name__ == "__main__":
    main()
