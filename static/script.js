document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const predictBtn = document.getElementById('predict-btn');
    const codeInput = document.getElementById('code-input');
    const loading = document.getElementById('loading');
    const resultArea = document.getElementById('result-area');

    let currentFileContent = '';

    // Tab Switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
        });
    });

    // File Upload Handling
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-color)';
        dropZone.style.background = 'rgba(0,0,0,0.3)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--glass-border)';
        dropZone.style.background = 'rgba(0,0,0,0.1)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--glass-border)';
        dropZone.style.background = 'rgba(0,0,0,0.1)';

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        fileInfo.textContent = `Selected: ${file.name}`;
        const reader = new FileReader();
        reader.onload = (e) => {
            currentFileContent = e.target.result;
        };
        reader.readAsText(file);
    }

    // Prediction Logic
    predictBtn.addEventListener('click', async () => {
        const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
        const language = document.querySelector('input[name="language"]:checked').value;
        let codeToAnalyze = '';

        if (activeTab === 'paste') {
            codeToAnalyze = codeInput.value;
        } else {
            codeToAnalyze = currentFileContent;
        }

        if (!codeToAnalyze.trim()) {
            alert('Please provide some code to analyze.');
            return;
        }

        // UI Updates
        predictBtn.disabled = true;
        loading.classList.remove('hidden');
        resultArea.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: codeToAnalyze,
                    language: language
                })
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                alert(`Error: ${data.error}`);
            }

        } catch (error) {
            alert('An error occurred while connecting to the server.');
            console.error(error);
        } finally {
            predictBtn.disabled = false;
            loading.classList.add('hidden');
        }
    });

    function showResult(data) {
        resultArea.classList.remove('hidden');
        const verdict = document.getElementById('verdict');
        const verdictDesc = document.getElementById('verdict-desc');
        const badge = document.getElementById('confidence-badge');
        const metricsGrid = document.getElementById('metrics-grid');

        // Update Verdict
        if (data.bug) {
            verdict.className = 'verdict bug';
            verdict.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> <span>Defect Detected</span>';
            verdictDesc.textContent = 'The model has identified patterns associated with software bugs.';
        } else {
            verdict.className = 'verdict clean';
            verdict.innerHTML = '<i class="fa-solid fa-check-circle"></i> <span>Clean Code</span>';
            verdictDesc.textContent = 'No significant defects detected based on software metrics.';
        }

        // Update Confidence
        const percentage = (data.probability * 100).toFixed(1);
        badge.textContent = `${percentage}% Probability`;

        // Update Metrics
        metricsGrid.innerHTML = '';
        const keyMetrics = ['loc', 'v(g)', 'n', 'd', 'e', 'b'];
        const labels = {
            'loc': 'Lines of Code',
            'v(g)': 'Complexity',
            'n': 'Length',
            'd': 'Difficulty',
            'e': 'Effort',
            'b': 'Est. Bugs'
        };

        keyMetrics.forEach(key => {
            if (data.metrics[key] !== undefined) {
                const div = document.createElement('div');
                div.className = 'metric-item';
                div.innerHTML = `
                    <span class="metric-label">${labels[key]}</span>
                    <span class="metric-value">${typeof data.metrics[key] === 'number' ? data.metrics[key].toFixed(2) : data.metrics[key]}</span>
                `;
                metricsGrid.appendChild(div);
            }
        });
    }
});
