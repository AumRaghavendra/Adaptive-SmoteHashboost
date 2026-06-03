document.addEventListener('DOMContentLoaded', () => {
    fetchAvailableDatasets();
    fetchResults();

    document.getElementById('run-btn').addEventListener('click', async () => {
        const select = document.getElementById('dataset-select');
        const selectedPath = select.value;
        if (!selectedPath) {
            alert('Please select a dataset to run!');
            return;
        }

        const runBtn = document.getElementById('run-btn');
        const loader = document.getElementById('global-loading');
        
        runBtn.disabled = true;
        runBtn.innerText = 'EXECUTING...';
        loader.classList.remove('hidden');

        try {
            const response = await fetch('/api/run_pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_path: selectedPath })
            });
            const data = await response.json();
            
            if (data.success) {
                // Instantly parse the results folder to grab the newly finished dataset output
                await fetchResults(selectedPath);
            } else {
                alert('Pipeline Error: ' + data.error);
            }
        } catch (err) {
            alert('Connection error running pipeline.');
        } finally {
            runBtn.disabled = false;
            runBtn.innerText = 'RUN PIPELINE';
            loader.classList.add('hidden');
        }
    });
});

async function fetchAvailableDatasets() {
    try {
        const response = await fetch('/api/available_datasets');
        const data = await response.json();
        
        const select = document.getElementById('dataset-select');
        select.innerHTML = '<option value="">-- Select a Dataset to Execute --</option>';
        
        data.datasets.forEach(path => {
            const opt = document.createElement('option');
            opt.value = path;
            opt.textContent = path;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to load dataset list:", e);
    }
}

async function fetchResults(targetBaseName = null) {
    const container = document.getElementById('results-container');
    
    try {
        const response = await fetch('/api/results');
        const data = await response.json();
        
        let targetDataset = null;
        let datasetsToRender = data.datasets;
        if (targetBaseName) {
            targetDataset = data.datasets.find(d => targetBaseName.includes(d.name) || d.name.includes(targetBaseName));
            if (targetDataset) {
                // Pin newly generated target to the very top, push the rest down
                datasetsToRender = [
                    targetDataset,
                    ...datasetsToRender.filter(d => d.name !== targetDataset.name)
                ];
            }
        }

        if (datasetsToRender.length === 0) {
            container.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 4rem;">
                    <h2 style="color: var(--text-secondary);">No results found.</h2>
                    <p style="margin-top: 1rem;">Select a dataset and click RUN PIPELINE.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = ''; // Clear loading
        
        datasetsToRender.forEach(dataset => {
            const card = document.createElement('div');
            card.className = 'dataset-card';
            
            const renderMetric = (label, key) => {
                const mean = dataset.metrics[`HashBoost_${key}`];
                const std = dataset.metrics[`HashBoost_${key}_std`];
                if (mean === undefined) return '';
                
                return `
                    <div class="metric-item">
                        <span class="metric-label">${label}</span>
                        <div class="metric-value">
                            ${mean.toFixed(4)}
                            <span class="metric-std">±${(std || 0).toFixed(4)}</span>
                        </div>
                    </div>
                `;
            };

            const metricsHtml = `
                <div class="metrics-grid">
                    ${renderMetric('F1 Score', 'F1')}
                    ${renderMetric('ROC AUC', 'AUC')}
                    ${renderMetric('G-Mean', 'GMean')}
                    ${renderMetric('Avg Precision', 'AP')}
                </div>
            `;

            const timestamp = Date.now();
            const plotHtml = dataset.has_plot ? `
                <div class="plot-container">
                    <img src="${dataset.plot_url}?t=${timestamp}" alt="${dataset.name} Metrics Plot">
                </div>
            ` : '';

            card.innerHTML = `
                <div class="card-header">
                    <h2>${dataset.name.toUpperCase()}</h2>
                </div>
                ${metricsHtml}
                ${plotHtml}
            `;
            
            container.appendChild(card);
        });

    } catch (error) {
        container.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 4rem; color: #ff6b6b;">
                <h2>Error loading metrics.</h2>
                <p>Please ensure the Flask backend is running and the output directory exists.</p>
            </div>
        `;
    }
}
