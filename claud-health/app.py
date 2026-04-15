"""
Flask frontend — healthcare treatments PPML.

  cd claud-health && python3 app.py
  → http://127.0.0.1:5050
"""
from __future__ import annotations

import io
import os
import sys

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)

import experiment_runner as er
import web_present as wp

app = Flask(
    __name__,
    template_folder=os.path.join(_PKG, "templates"),
    static_folder=os.path.join(_PKG, "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-healthcare-ppml-key")


def _output_dir() -> str:
    return er.default_output_dir()


def _safe_plot_name(name: str) -> bool:
    base = os.path.basename(name)
    return base == name and name.lower().endswith(".png") and ".." not in name


@app.route("/")
def index():
    import pandas as pd

    csv_path = os.path.join(_output_dir(), "metrics_all_runs.csv")
    has_results = os.path.isfile(csv_path)
    if has_results:
        df = pd.read_csv(csv_path)
        meta = wp.load_run_metadata(_output_dir(), df=df)
    else:
        meta = {}
    return render_template(
        "index.html",
        has_results=has_results,
        epsilons=meta.get("epsilons", er.DEFAULT_EPSILONS),
        delta_fmt=f"{float(meta.get('delta', er.DEFAULT_DELTA)):.2e}",
    )


@app.route("/run", methods=["POST"])
def run_experiment_route():
    log_buf = io.StringIO()
    try:
        er.run_experiment(verbose=False, log_print=log_buf)
        flash("Experiment finished successfully.", "success")
    except Exception as exc:
        flash(f"Run failed: {exc}", "error")
        return redirect(url_for("index"))
    return redirect(url_for("results"))


@app.route("/results")
def results():
    import pandas as pd

    csv_path = os.path.join(_output_dir(), "metrics_all_runs.csv")
    if not os.path.isfile(csv_path):
        flash("No results yet. Run the experiment first.", "warning")
        return redirect(url_for("index"))

    df = pd.read_csv(csv_path)
    out_dir = _output_dir()
    page = wp.build_page_context(out_dir, df)

    plot_urls = [
        url_for("plots", filename=f)
        for f in er.PLOT_FILENAMES
        if os.path.isfile(os.path.join(out_dir, f))
    ]

    best = df.loc[df["accuracy"].idxmax()] if len(df) else None
    baselines = df[df["stage"] == "BASELINE"] if "stage" in df.columns else None

    return render_template(
        "results.html",
        n_rows=len(df),
        best=best,
        baselines=baselines,
        plot_urls=plot_urls,
        **page,
    )


@app.route("/plots/<path:filename>")
def plots(filename: str):
    if not _safe_plot_name(filename):
        abort(404)
    return send_from_directory(_output_dir(), filename, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=True)
