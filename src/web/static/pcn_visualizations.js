// PCN Visualization Functions

let pcnComparisonChart = null;
let pcnEnergyChart = null;
let pcnDiversityChart = null;
let hybridPerformanceChart = null;

// Initialize PCN experiment charts
function initPCNCharts() {
    // Initialize comparison chart for data leakage experiment
    const comparisonCtx = document.getElementById('pcn-comparison-chart');
    if (comparisonCtx && !pcnComparisonChart) {
        pcnComparisonChart = new Chart(comparisonCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'With Label Leakage (Problematic)',
                    data: [],
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }, {
                    label: 'Without Label Leakage (Correct)',
                    data: [],
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Initialize energy distribution chart
    const energyCtx = document.getElementById('pcn-energy-chart');
    if (energyCtx && !pcnEnergyChart) {
        pcnEnergyChart = new Chart(energyCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Energy',
                    data: [],
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Initialize diversity scores chart
    const diversityCtx = document.getElementById('pcn-diversity-chart');
    if (diversityCtx && !pcnDiversityChart) {
        pcnDiversityChart = new Chart(diversityCtx.getContext('2d'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Diversity Score',
                    data: [],
                    backgroundColor: 'rgba(236, 72, 153, 0.5)',
                    borderColor: 'rgb(236, 72, 153)',
                    pointRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Sample Index'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Diversity Score'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Initialize hybrid model charts
function initHybridArchitecture() {
    const performanceCtx = document.getElementById('hybrid-performance-chart');
    if (performanceCtx && !hybridPerformanceChart) {
        hybridPerformanceChart = new Chart(performanceCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Hybrid Model Loss',
                    data: [],
                    borderColor: 'rgb(139, 92, 246)',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.1
                }, {
                    label: 'Standard Transformer Loss',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }
    
    // Draw initial architecture diagram
    drawArchitectureDiagram('pcn-ff');
}

// Update PCN comparison chart
function updatePCNComparison(withLeakage, withoutLeakage, epoch = null) {
    if (pcnComparisonChart) {
        // Add new data point
        if (epoch !== null) {
            pcnComparisonChart.data.labels.push(`Epoch ${epoch + 1}`);
        } else {
            const currentLength = pcnComparisonChart.data.labels.length;
            pcnComparisonChart.data.labels.push(`Step ${currentLength + 1}`);
        }
        
        pcnComparisonChart.data.datasets[0].data.push(withLeakage);
        pcnComparisonChart.data.datasets[1].data.push(withoutLeakage);
        
        // Keep only last 10 points for clarity
        if (pcnComparisonChart.data.labels.length > 10) {
            pcnComparisonChart.data.labels.shift();
            pcnComparisonChart.data.datasets[0].data.shift();
            pcnComparisonChart.data.datasets[1].data.shift();
        }
        
        pcnComparisonChart.update();
    }
}

// Update PCN energy chart
function updatePCNEnergy(energyData) {
    if (pcnEnergyChart) {
        pcnEnergyChart.data.labels = energyData.steps;
        pcnEnergyChart.data.datasets[0].data = energyData.values;
        pcnEnergyChart.update();
    }
}

// Update PCN diversity chart
function updatePCNDiversity(diversityData) {
    if (pcnDiversityChart) {
        const scatterData = diversityData.map((value, index) => ({
            x: index,
            y: value
        }));
        pcnDiversityChart.data.datasets[0].data = scatterData;
        pcnDiversityChart.update();
    }
}

// Update hybrid performance chart
function updateHybridPerformance(hybridLoss, baselineLoss) {
    if (hybridPerformanceChart) {
        // Add new data point
        const label = hybridPerformanceChart.data.labels.length;
        hybridPerformanceChart.data.labels.push(label);
        hybridPerformanceChart.data.datasets[0].data.push(hybridLoss);
        hybridPerformanceChart.data.datasets[1].data.push(baselineLoss);
        
        // Keep only last 100 points
        if (hybridPerformanceChart.data.labels.length > 100) {
            hybridPerformanceChart.data.labels.shift();
            hybridPerformanceChart.data.datasets[0].data.shift();
            hybridPerformanceChart.data.datasets[1].data.shift();
        }
        
        hybridPerformanceChart.update('none');
    }
}

// Draw architecture diagram
function drawArchitectureDiagram(architecture) {
    const diagramDiv = document.getElementById('hybrid-arch-diagram');
    if (!diagramDiv) return;
    
    // Simple text-based diagrams for now
    const diagrams = {
        'pcn-ff': `
            <div class="arch-diagram">
                <div class="arch-block">Input</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block">Embedding</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block highlight">Attention</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block pcn">PCN Layer (replaces FFN)</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block">Output</div>
            </div>
        `,
        'alternating': `
            <div class="arch-diagram">
                <div class="arch-block">Input</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block highlight">Attention Layer</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block pcn">PCN Layer</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block highlight">Attention Layer</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block pcn">PCN Layer</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block">Output</div>
            </div>
        `,
        'hierarchical': `
            <div class="arch-diagram">
                <div class="arch-block">Input</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block pcn">PCN Feature Extractor</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block highlight">Transformer Stack</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block">Output</div>
            </div>
        `,
        'dual-stream': `
            <div class="arch-diagram dual-stream">
                <div class="stream">
                    <div class="arch-block">Input</div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-block pcn">PCN Stream</div>
                </div>
                <div class="merge">→ ⊕ ←</div>
                <div class="stream">
                    <div class="arch-block">Input</div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-block highlight">Transformer Stream</div>
                </div>
                <div class="arch-arrow full-width">↓</div>
                <div class="arch-block">Merged Output</div>
            </div>
        `,
        'pcn-positional': `
            <div class="arch-diagram">
                <div class="arch-block">Input</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block pcn">PCN Positional Encoding</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block">Embedding + Adaptive Position</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block highlight">Standard Transformer</div>
                <div class="arch-arrow">↓</div>
                <div class="arch-block">Output</div>
            </div>
        `
    };
    
    diagramDiv.innerHTML = diagrams[architecture] || '<p>Architecture diagram not available</p>';
}

// Add styles for architecture diagrams
const style = document.createElement('style');
style.textContent = `
    .arch-diagram {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        font-family: monospace;
    }
    
    .arch-diagram.dual-stream {
        flex-direction: row;
        justify-content: center;
        align-items: flex-start;
    }
    
    .arch-block {
        background-color: #e5e7eb;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 5px;
        font-weight: bold;
        text-align: center;
        min-width: 150px;
    }
    
    .arch-block.highlight {
        background-color: #dbeafe;
        border: 2px solid #3b82f6;
    }
    
    .arch-block.pcn {
        background-color: #ede9fe;
        border: 2px solid #8b5cf6;
    }
    
    .arch-arrow {
        font-size: 20px;
        color: #6b7280;
        margin: 5px;
    }
    
    .arch-arrow.full-width {
        width: 100%;
        text-align: center;
    }
    
    .stream {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 20px;
    }
    
    .merge {
        display: flex;
        align-items: center;
        font-size: 24px;
        margin: 0 10px;
        height: 100%;
    }
`;
document.head.appendChild(style);

// Bio-plausibility function removed - not meaningful for this project

// PCN Experiment Event Handlers
document.addEventListener('DOMContentLoaded', () => {
    // PCN experiment controls
    const startPCNButton = document.getElementById('start-pcn-experiment');
    if (startPCNButton) {
        startPCNButton.addEventListener('click', startPCNExperiment);
    }
    
    const stopPCNButton = document.getElementById('stop-pcn-experiment');
    if (stopPCNButton) {
        stopPCNButton.addEventListener('click', stopPCNExperiment);
    }
    
    // Hybrid training controls
    const startHybridButton = document.getElementById('start-hybrid-training');
    if (startHybridButton) {
        startHybridButton.addEventListener('click', startHybridTraining);
    }
    
    const stopHybridButton = document.getElementById('stop-hybrid-training');
    if (stopHybridButton) {
        stopHybridButton.addEventListener('click', stopHybridTraining);
    }
    
    // Slider updates
    const pcnSliders = [
        { id: 'pcn-samples', valueId: 'pcn-samples-value' },
        { id: 'pcn-refine-steps', valueId: 'pcn-refine-value' },
        { id: 'pcn-noise-scale', valueId: 'pcn-noise-value' },
        { id: 'pcn-learning-rate', valueId: 'pcn-lr-value' },
        { id: 'pcn-inference-steps', valueId: 'pcn-inference-value' },
        { id: 'pcn-energy-threshold', valueId: 'pcn-threshold-value' }
    ];
    
    pcnSliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        if (element) {
            element.addEventListener('input', (e) => {
                document.getElementById(slider.valueId).textContent = e.target.value;
            });
        }
    });
});

// Start PCN experiment
async function startPCNExperiment() {
    const config = {
        enable_leakage: 'both', // Run both experiments
        num_samples: parseInt(document.getElementById('pcn-samples').value),
        refine_steps: parseInt(document.getElementById('pcn-refine-steps').value),
        noise_scale: parseFloat(document.getElementById('pcn-noise-scale').value)
    };
    
    // Reset all charts
    if (pcnComparisonChart) {
        pcnComparisonChart.data.labels = [];
        pcnComparisonChart.data.datasets[0].data = [];
        pcnComparisonChart.data.datasets[1].data = [];
        pcnComparisonChart.update();
    }
    
    if (pcnEnergyChart) {
        pcnEnergyChart.data.labels = [];
        pcnEnergyChart.data.datasets[0].data = [];
        pcnEnergyChart.update();
    }
    
    if (pcnDiversityChart) {
        pcnDiversityChart.data.datasets[0].data = [];
        pcnDiversityChart.update();
    }
    
    // Reset output displays
    document.getElementById('pcn-output-leaked').innerHTML = '<p class="output-placeholder">Experiment running...</p>';
    document.getElementById('pcn-output-clean').innerHTML = '<p class="output-placeholder">Experiment running...</p>';
    
    try {
        const response = await fetch('/api/pcn/start-experiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            console.log('PCN experiment started');
            // Update UI to show experiment is running
            document.getElementById('start-pcn-experiment').disabled = true;
            document.getElementById('stop-pcn-experiment').disabled = false;
        }
    } catch (error) {
        console.error('Failed to start PCN experiment:', error);
    }
}

// Stop PCN experiment
async function stopPCNExperiment() {
    try {
        const response = await fetch('/api/pcn/stop-experiment', {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log('PCN experiment stopped');
            document.getElementById('start-pcn-experiment').disabled = false;
            document.getElementById('stop-pcn-experiment').disabled = true;
        }
    } catch (error) {
        console.error('Failed to stop PCN experiment:', error);
    }
}

// Start hybrid training
async function startHybridTraining() {
    const architecture = document.getElementById('hybrid-architecture').value;
    const config = {
        architecture: architecture,
        pcn_lr: parseFloat(document.getElementById('pcn-learning-rate').value),
        pcn_steps: parseInt(document.getElementById('pcn-inference-steps').value),
        energy_threshold: parseFloat(document.getElementById('pcn-energy-threshold').value),
        use_exploration: document.getElementById('use-pcn-exploration').checked
    };
    
    try {
        const response = await fetch('/api/hybrid/start-training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            console.log('Hybrid training started');
            document.getElementById('start-hybrid-training').disabled = true;
            document.getElementById('stop-hybrid-training').disabled = false;
        }
    } catch (error) {
        console.error('Failed to start hybrid training:', error);
    }
}

// Stop hybrid training
async function stopHybridTraining() {
    try {
        const response = await fetch('/api/hybrid/stop-training', {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log('Hybrid training stopped');
            document.getElementById('start-hybrid-training').disabled = false;
            document.getElementById('stop-hybrid-training').disabled = true;
        }
    } catch (error) {
        console.error('Failed to stop hybrid training:', error);
    }
}