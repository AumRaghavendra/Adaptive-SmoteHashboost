import os
import glob
import subprocess
from flask import Flask, render_template, jsonify, send_from_directory, request
import pandas as pd

import shutil

app = Flask(__name__)
RESULTS_DIR = "my_new_visualized_results"

if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

import sys
import time

@app.route("/api/available_datasets")
def available_datasets():
    csv_files = glob.glob("**/*.csv", recursive=True) + glob.glob("**/*.data", recursive=True)
    
    # Strictly filter out system clutter and generated results
    valid_datasets = []
    for f in csv_files:
        path_lower = f.lower()
        if "venv" in path_lower or "env" in path_lower or RESULTS_DIR.lower() in path_lower:
            continue
        if "_balanced" in path_lower or "_metrics" in path_lower:
            continue
            
        # JS JSON.stringify destroys isolated Windows backslashes (\) leading to Python os.path.exists failures
        # Convert path to UNIX forward slashes so HTTP transport is flawless
        valid_datasets.append(f.replace('\\', '/'))
        
    return jsonify({"datasets": sorted(valid_datasets)})

@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline():
    data = request.json
    dataset_path = data.get("dataset_path")
    if not dataset_path:
        return jsonify({"error": "No dataset selected"}), 400
        
    try:
        # Construct the execution command securely inheriting the correct Python runtime!
        subprocess.run([sys.executable, "solution.py", "--input", dataset_path, "--output", RESULTS_DIR], check=True)
        return jsonify({"success": True, "message": "Pipeline completed securely!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/results")
def get_results():
    if not os.path.exists(RESULTS_DIR):
        return jsonify({"datasets": []})
        
    metrics_files = glob.glob(os.path.join(RESULTS_DIR, "*_metrics.csv"))
    datasets = []
    
    for mf in metrics_files:
        base_name = os.path.basename(mf).replace('_metrics.csv', '')
        
        # Load metrics
        try:
            df = pd.read_csv(mf)
            metrics_data = df.to_dict('records')[0] if not df.empty else {}
        except Exception:
            metrics_data = {}
            
        # Check if plot exists
        plot_path = os.path.join(RESULTS_DIR, f"{base_name}_plot.png")
        has_plot = os.path.exists(plot_path)
        
        datasets.append({
            "name": base_name,
            "metrics": metrics_data,
            "has_plot": has_plot,
            "plot_url": f"/results/{base_name}_plot.png?t={int(time.time())}" if has_plot else None
        })
        
    return jsonify({"datasets": datasets})

@app.route("/results/<path:filename>")
def serve_result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
