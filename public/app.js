// Global State
let selectedDataset = 'dataset_1';
let metricsData = null;
let shapData = null;
let isPredictFS = false;

// Chart Instances
let r2ChartInstance = null;
let errorChartInstance = null;
let shapChartInstance = null;

// Dataset feature structures
const datasetFeatures = {
    dataset_1: [
        { name: 'Hours Studied', type: 'slider', min: 1, max: 9, step: 1, default: 5, fs: true },
        { name: 'Previous Scores', type: 'slider', min: 40, max: 99, step: 1, default: 70, fs: true },
        { name: 'Extracurricular Activities', type: 'select', options: ['No', 'Yes'], default: 'No', fs: false },
        { name: 'Sleep Hours', type: 'slider', min: 4, max: 9, step: 1, default: 7, fs: false },
        { name: 'Sample Question Papers Practiced', type: 'slider', min: 0, max: 9, step: 1, default: 4, fs: false }
    ],
    dataset_2: [
        { name: 'Hours_Studied', label: 'Hours Studied', type: 'slider', min: 1, max: 30, step: 1, default: 10, fs: true },
        { name: 'Attendance', label: 'Attendance (%)', type: 'slider', min: 60, max: 100, step: 1, default: 90, fs: true },
        { name: 'Sleep_Hours', label: 'Sleep Hours', type: 'slider', min: 4, max: 9, step: 1, default: 7, fs: false },
        { name: 'Previous_Scores', label: 'Previous Scores', type: 'slider', min: 40, max: 99, step: 1, default: 70, fs: true },
        { name: 'Tutoring_Sessions', label: 'Tutoring Sessions', type: 'slider', min: 0, max: 10, step: 1, default: 2, fs: false },
        { name: 'Physical_Activity', label: 'Physical Activity (hrs/wk)', type: 'slider', min: 0, max: 8, step: 1, default: 3, fs: false },
        { name: 'Gender', type: 'select', options: ['Female', 'Male'], default: 'Female', fs: false },
        { name: 'Parental_Involvement', label: 'Parental Involvement', type: 'select', options: ['Low', 'Medium', 'High'], default: 'Medium', fs: false },
        { name: 'Access_to_Resources', label: 'Access to Resources', type: 'select', options: ['Low', 'Medium', 'High'], default: 'Medium', fs: true },
        { name: 'Extracurricular_Activities', label: 'Extracurricular Activities', type: 'select', options: ['No', 'Yes'], default: 'No', fs: false },
        { name: 'Motivation_Level', label: 'Motivation Level', type: 'select', options: ['Low', 'Medium', 'High'], default: 'Medium', fs: false },
        { name: 'Internet_Access', label: 'Internet Access', type: 'select', options: ['No', 'Yes'], default: 'Yes', fs: false },
        { name: 'Family_Income', label: 'Family Income', type: 'select', options: ['Low', 'Medium', 'High'], default: 'Medium', fs: false },
        { name: 'Teacher_Quality', label: 'Teacher Quality', type: 'select', options: ['Low', 'Medium', 'High'], default: 'Medium', fs: false },
        { name: 'School_Type', label: 'School Type', type: 'select', options: ['Public', 'Private'], default: 'Public', fs: false },
        { name: 'Peer_Influence', label: 'Peer Influence', type: 'select', options: ['Negative', 'Neutral', 'Positive'], default: 'Neutral', fs: false },
        { name: 'Learning_Disabilities', label: 'Learning Disabilities', type: 'select', options: ['No', 'Yes'], default: 'No', fs: false },
        { name: 'Parental_Education_Level', label: 'Parental Education', type: 'select', options: ['High School', 'Associate\'s', 'Bachelor\'s', 'Postgraduate'], default: 'Bachelor\'s', fs: false },
        { name: 'Distance_from_Home', label: 'Distance from Home', type: 'select', options: ['Near', 'Moderate', 'Far'], default: 'Near', fs: false }
    ]
};

// UI Element References
const sectionMetrics = document.getElementById('section-metrics');
const sectionPredict = document.getElementById('section-predict');
const sectionExplain = document.getElementById('section-explain');
const globalDatasetSelect = document.getElementById('global-dataset-select');

// Tab Switcher
function switchTab(tabName) {
    // Toggle active nav item
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${tabName}`).classList.add('active');

    // Toggle active section
    sectionMetrics.classList.remove('active');
    sectionPredict.classList.remove('active');
    sectionExplain.classList.remove('active');

    if (tabName === 'metrics') {
        sectionMetrics.classList.add('active');
        renderMetricsVisuals();
    } else if (tabName === 'predict') {
        sectionPredict.classList.add('active');
        generatePredictForm();
    } else if (tabName === 'explain') {
        sectionExplain.classList.add('active');
        renderShapVisuals();
    }
}

// Fetch Metrics from backend
async function fetchMetricsData() {
    try {
        const [metricsRes, shapRes] = await Promise.all([
            fetch('/api/metrics'),
            fetch('/api/shap')
        ]);
        metricsData = await metricsRes.json();
        shapData = await shapRes.json();
        renderMetricsVisuals();
        renderShapVisuals();
    } catch (err) {
        console.error("Failed to load metrics: ", err);
        const tableBody = document.getElementById('metrics-table-body');
        tableBody.innerHTML = `<tr><td colspan="5" class="text-center text-rose" style="color: var(--color-rose);"><i class="fa-solid fa-triangle-exclamation"></i> Error loading metrics from server. Make sure the Flask API backend is running.</td></tr>`;
    }
}

// On global dataset select change
function onDatasetChange(val) {
    selectedDataset = val;
    renderMetricsVisuals();
    renderShapVisuals();
    generatePredictForm();
    
    // Update SHAP text
    const shapText = document.getElementById('shap-insight-text');
    if (selectedDataset === 'dataset_1') {
        shapText.innerHTML = `For Dataset 1, <strong>Previous Scores</strong> accounts for approximately 62% of the model's overall prediction power, followed by <strong>Hours Studied</strong>. Other variables like Sleep Hours or extracurricular activities show negligible global influence.`;
    } else {
        shapText.innerHTML = `For Dataset 2, <strong>Attendance</strong> and <strong>Hours Studied</strong> show the highest global feature importances, consistent with their strong correlations ($r=0.58$ and $r=0.45$). Family income and teacher quality exert moderate influence, while peer influence and school type have minimal global impact.`;
    }
}

// Render metrics, charts, and table
function renderMetricsVisuals() {
    if (!metricsData) return;

    const dataKey = selectedDataset === 'dataset_1' ? 'dataset_1_all' : 'dataset_2_all';
    const activeData = metricsData[dataKey];
    if (!activeData || !activeData["Ensemble VR"]) return;

    // 1. Update Cards
    const vrMetrics = activeData["Ensemble VR"];
    document.getElementById('val-r2').innerText = vrMetrics.R2.toFixed(4);
    document.getElementById('val-rmse').innerText = vrMetrics.RMSE.toFixed(4);
    document.getElementById('val-mae').innerText = vrMetrics.MAE.toFixed(4);
    document.getElementById('lbl-r2-sub').innerText = "Top 5 Models Voting Ensemble";

    // 2. Render comparative table
    renderMetricsTable();

    // 3. Render charts
    renderR2Chart(false);
    renderErrorChart(false);
}

// Render metrics table
function renderMetricsTable() {
    const isFS = document.getElementById('btn-chart-r2-fs').classList.contains('active');
    const dataKey = `${selectedDataset}_${isFS ? 'fs' : 'all'}`;
    const data = metricsData[dataKey];
    if (!data) return;

    document.getElementById('lbl-table-subset').innerText = isFS ? "Feature Selected Subset" : "All Features";

    const tbody = document.getElementById('metrics-table-body');
    tbody.innerHTML = '';

    // Order algorithms logically
    const order = [
        "Linear Regression", "Ridge", "SVR", "CatBoosting Regressor", "XGBRegressor", 
        "Random Forest Regressor", "K-Neighbors Regressor", "Bagging Regressor", "AdaBoost Regressor", 
        "Ensemble VR"
    ];

    order.forEach(name => {
        if (data[name]) {
            const tr = document.createElement('tr');
            if (name === 'Ensemble VR') {
                tr.className = 'highlight-row';
            }
            
            const cvVal = (name === 'Ensemble VR' && data[name].CV_R2_Mean) 
                ? data[name].CV_R2_Mean.toFixed(4) 
                : '-';

            tr.innerHTML = `
                <td><strong>${name}</strong></td>
                <td>${data[name].MAE.toFixed(4)}</td>
                <td>${data[name].RMSE.toFixed(4)}</td>
                <td>${data[name].R2.toFixed(4)}</td>
                <td><strong>${cvVal}</strong></td>
            `;
            tbody.appendChild(tr);
        }
    });
}

// Chart Toggle Mode
function changeChartMode(chartType, isFS) {
    if (chartType === 'r2') {
        document.getElementById('btn-chart-r2-all').classList.toggle('active', !isFS);
        document.getElementById('btn-chart-r2-fs').classList.toggle('active', isFS);
        renderR2Chart(isFS);
    } else {
        document.getElementById('btn-chart-err-all').classList.toggle('active', !isFS);
        document.getElementById('btn-chart-err-fs').classList.toggle('active', isFS);
        renderErrorChart(isFS);
    }
    renderMetricsTable();
}

// Render R2 comparison chart
function renderR2Chart(isFS) {
    const ctx = document.getElementById('r2-comparison-chart').getContext('2d');
    const dataKey = `${selectedDataset}_${isFS ? 'fs' : 'all'}`;
    const data = metricsData[dataKey];
    if (!data) return;

    const models = Object.keys(data);
    const r2Scores = models.map(m => data[m].R2);

    if (r2ChartInstance) {
        r2ChartInstance.destroy();
    }

    // Set colors (accentuate Ensemble VR)
    const colors = models.map(m => m === 'Ensemble VR' ? '#06b6d4' : 'rgba(255, 255, 255, 0.25)');
    const borderColors = models.map(m => m === 'Ensemble VR' ? '#06b6d4' : 'rgba(255, 255, 255, 0.4)');

    r2ChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'R² Score',
                data: r2Scores,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' },
                    min: Math.max(0, Math.min(...r2Scores) - 0.05)
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            }
        }
    });
}

// Render MAE / RMSE grouped bar chart
function renderErrorChart(isFS) {
    const ctx = document.getElementById('error-comparison-chart').getContext('2d');
    const dataKey = `${selectedDataset}_${isFS ? 'fs' : 'all'}`;
    const data = metricsData[dataKey];
    if (!data) return;

    const models = Object.keys(data);
    const maeScores = models.map(m => data[m].MAE);
    const rmseScores = models.map(m => data[m].RMSE);

    if (errorChartInstance) {
        errorChartInstance.destroy();
    }

    errorChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'MAE',
                    data: maeScores,
                    backgroundColor: 'rgba(6, 182, 212, 0.65)',
                    borderColor: '#06b6d4',
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: 'RMSE',
                    data: rmseScores,
                    backgroundColor: 'rgba(168, 85, 247, 0.65)',
                    borderColor: '#a855f7',
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    display: true, 
                    labels: { color: '#94a3b8' } 
                }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            }
        }
    });
}

// Render Global SHAP bar chart
function renderShapVisuals() {
    if (!shapData) return;

    const ctx = document.getElementById('shap-global-chart').getContext('2d');
    const data = shapData[selectedDataset];
    if (!data) return;

    const features = Object.keys(data);
    const importances = Object.values(data);

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Mean Absolute SHAP Value (Impact on prediction)',
                data: importances,
                backgroundColor: 'rgba(168, 85, 247, 0.7)',
                borderColor: '#a855f7',
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 12, weight: 'bold' } }
                }
            }
        }
    });
}

// Generate the forms dynamically based on selected dataset
function generatePredictForm() {
    const container = document.getElementById('prediction-inputs-container');
    container.innerHTML = '';

    const features = datasetFeatures[selectedDataset];
    
    features.forEach(feat => {
        const div = document.createElement('div');
        div.className = 'form-group';
        div.id = `fg-${feat.name}`;

        const labelName = feat.label || feat.name;
        
        if (feat.type === 'slider') {
            div.innerHTML = `
                <label for="inp-${feat.name}">
                    <span>${labelName}</span>
                    <span class="slider-val" id="val-lbl-${feat.name}">${feat.default}</span>
                </label>
                <input type="range" id="inp-${feat.name}" name="${feat.name}" 
                       min="${feat.min}" max="${feat.max}" step="${feat.step}" 
                       value="${feat.default}" oninput="document.getElementById('val-lbl-${feat.name}').innerText = this.value">
            `;
        } else if (feat.type === 'select') {
            let optionsHtml = '';
            feat.options.forEach(opt => {
                optionsHtml += `<option value="${opt}" ${opt === feat.default ? 'selected' : ''}>${opt}</option>`;
            });
            div.innerHTML = `
                <label for="inp-${feat.name}"><span>${labelName}</span></label>
                <select id="inp-${feat.name}" name="${feat.name}">
                    ${optionsHtml}
                </select>
            `;
        }
        
        container.appendChild(div);
    });

    togglePredictFeatureSelection(isPredictFS);
}

// Disable/Enable features depending on feature selection mode
function togglePredictFeatureSelection(isFS) {
    isPredictFS = isFS;
    const features = datasetFeatures[selectedDataset];
    
    features.forEach(feat => {
        const element = document.getElementById(`inp-${feat.name}`);
        const group = document.getElementById(`fg-${feat.name}`);
        if (!element) return;

        if (isFS && !feat.fs) {
            element.disabled = true;
            group.style.opacity = '0.35';
            group.style.pointerEvents = 'none';
        } else {
            element.disabled = false;
            group.style.opacity = '1';
            group.style.pointerEvents = 'all';
        }
    });
}

// Request Prediction & explanation from backend API
async function requestPrediction() {
    const features = datasetFeatures[selectedDataset];
    const inputs = {};

    features.forEach(feat => {
        const element = document.getElementById(`inp-${feat.name}`);
        if (element && !element.disabled) {
            inputs[feat.name] = element.value;
        } else if (element) {
            // If disabled by feature selection, pass the default value or a fallback
            inputs[feat.name] = feat.default;
        }
    });

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: selectedDataset,
                feature_selection: isPredictFS,
                features: inputs
            })
        });
        
        const result = await response.json();
        if (result.error) {
            alert("Error: " + result.error);
            return;
        }

        updatePredictionGauge(result.prediction);
        renderLimeExplanation(result.lime);
    } catch (err) {
        console.error("Prediction request failed:", err);
        alert("Failed to connect to backend server for prediction.");
    }
}

// Update the circular progress gauge
function updatePredictionGauge(score) {
    const fillCircle = document.getElementById('gauge-fill-circle');
    const lblVal = document.getElementById('lbl-prediction-value');
    const lblDesc = document.getElementById('lbl-prediction-description');

    // Round
    const rounded = score.toFixed(1);
    lblVal.innerText = rounded;

    // Dash offset calculation: dasharray is 282.7 (100% of circle circumference)
    // We want the gauge to represent score between 0 and 100
    const percent = Math.min(100, Math.max(0, score));
    const offset = 282.7 - (282.7 * percent / 100);
    fillCircle.style.strokeDashoffset = offset;

    // Update text description
    let desc = "";
    if (selectedDataset === 'dataset_1') {
        desc = `Performance Index estimated at <strong>${rounded}%</strong> using Voting Ensemble VR.`;
    } else {
        desc = `Exam Score estimated at <strong>${rounded}/100</strong>. `;
        if (score >= 80) desc += "Student is predicted to achieve <strong>Excellent</strong> performance.";
        else if (score >= 60) desc += "Student is predicted to achieve <strong>Satisfactory</strong> performance.";
        else desc += "Student is identified <strong>at-risk</strong> of low academic results.";
    }
    lblDesc.innerHTML = desc;
}

// Render LIME contributions bar charts
function renderLimeExplanation(limeData) {
    const container = document.getElementById('lime-bars-container');
    container.innerHTML = '';

    if (!limeData || limeData.length === 0) {
        container.innerHTML = '<div class="no-data-msg"><p>No explanation data returned.</p></div>';
        return;
    }

    // Find the max absolute contribution to normalize lengths for the bar widths
    const maxVal = Math.max(...limeData.map(item => Math.abs(item.contribution)));

    limeData.forEach(item => {
        const row = document.createElement('div');
        row.className = 'lime-bar-row';

        const directionClass = item.contribution >= 0 ? 'positive' : 'negative';
        const sign = item.contribution >= 0 ? '+' : '';
        
        // Width of the bar relative to the maximum contribution (percentage between 0 and 100)
        const barWidthPercent = maxVal > 0 ? (Math.abs(item.contribution) / maxVal * 100) : 0;

        row.innerHTML = `
            <div class="lime-bar-info">
                <span class="lime-rule">${item.rule}</span>
                <span class="lime-weight ${directionClass}">${sign}${item.contribution.toFixed(4)}</span>
            </div>
            <div class="lime-bar-outer">
                <div class="lime-bar-inner ${directionClass}" style="width: ${barWidthPercent}%"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    fetchMetricsData();
    generatePredictForm();
});
