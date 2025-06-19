let ws = null;
let lossChart = null;
let lrScheduleChart = null;
let currentConfig = {};
let architectureViz = null;
let animationsEnabled = true;

// Initialize WebSocket connection
function initWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;
    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected successfully');
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data.type);
            
            if (data.type === 'metrics') {
                updateMetrics(data.data);
            } else if (data.type === 'status') {
                updateStatus(data.data);
            } else if (data.type === 'training_complete') {
                handleTrainingComplete(data.data);
            } else if (data.type === 'activation_update') {
                updateActivations(data.data);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setTimeout(initWebSocket, 3000); // Reconnect after 3 seconds
    };
}

// Reset chart data
function resetChart() {
    if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update('none');
    }
    
    // Also hide any completion messages
    const completionDiv = document.getElementById('training-completion');
    if (completionDiv) {
        completionDiv.style.display = 'none';
    }
}

// Initialize loss chart
function initChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#4a9eff',
                backgroundColor: 'rgba(74, 158, 255, 0.1)',
                tension: 0.1
            }, {
                label: 'Validation Loss',
                data: [],
                borderColor: '#ff6384',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: '#444'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                },
                x: {
                    grid: {
                        color: '#444'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        }
    });
}

// Update metrics display
function updateMetrics(metrics) {
    console.log('Received metrics:', metrics);
    
    document.getElementById('current-step').textContent = metrics.step;
    document.getElementById('current-loss').textContent = metrics.train_loss.toFixed(4);
    document.getElementById('current-perplexity').textContent = metrics.perplexity.toFixed(2);
    
    // Update learning rate display
    if ('learning_rate' in metrics) {
        document.getElementById('current-lr').textContent = metrics.learning_rate.toExponential(4);
    }
    
    // Update epoch display
    if ('epoch' in metrics) {
        document.getElementById('current-epoch').textContent = metrics.epoch + 1;
    }
    
    // Update architecture visualization status in visualization mode
    if (currentConfig.training.visualization_mode && architectureViz) {
        architectureViz.updateTrainingStatus('Training', metrics.step);
    }
    
    // Update chart - show every 50 steps instead of 100 for more responsive updates
    if (lossChart && metrics.step % 50 === 0) {
        lossChart.data.labels.push(metrics.step);
        lossChart.data.datasets[0].data.push(metrics.train_loss);
        
        // Always add validation loss (even if it's the same as before) to keep lines continuous
        lossChart.data.datasets[1].data.push(metrics.val_loss);
        
        // Keep only last 50 points
        if (lossChart.data.labels.length > 50) {
            lossChart.data.labels.shift();
            lossChart.data.datasets[0].data.shift();
            lossChart.data.datasets[1].data.shift();
        }
        
        lossChart.update('none');
    }
    
    // Update LR schedule chart current position
    updateLRSchedulePosition(metrics.step);
}

// Update training status
function updateStatus(status) {
    const statusText = status.is_training ? 'Training' : 'Not Training';
    document.getElementById('training-status').textContent = statusText;
    document.getElementById('current-step').textContent = status.current_step;
    document.getElementById('total-steps').textContent = status.total_steps;
    
    // Update epoch info if available
    if ('current_epoch' in status && 'total_epochs' in status) {
        document.getElementById('current-epoch').textContent = status.current_epoch + 1; // 0-indexed
        document.getElementById('total-epochs').textContent = status.total_epochs;
    }
    
    // Update LR phase if scheduler info available
    if (status.scheduler_info && Object.keys(status.scheduler_info).length > 0) {
        updateLRPhase(status.scheduler_info);
    }
    
    // Enable/disable buttons based on training status
    document.getElementById('start-training').disabled = status.is_training;
    document.getElementById('stop-training').disabled = !status.is_training;
    
    // Update button styles
    if (status.is_training) {
        document.getElementById('start-training').classList.add('btn-disabled');
        document.getElementById('stop-training').classList.remove('btn-disabled');
    } else {
        document.getElementById('start-training').classList.remove('btn-disabled');
        document.getElementById('stop-training').classList.add('btn-disabled');
    }
}

// Load current configuration
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        currentConfig = await response.json();
        updateConfigUI(currentConfig);
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

// Update UI with configuration values
function updateConfigUI(config) {
    document.getElementById('use-layer-norm').checked = config.model.use_layer_norm;
    document.getElementById('use-residual').checked = config.model.use_residual;
    document.querySelector(`input[name="norm-position"][value="${config.model.norm_position}"]`).checked = true;
    
    document.getElementById('n-layers').value = config.model.n_layers;
    document.getElementById('n-layers-value').textContent = config.model.n_layers;
    
    document.getElementById('n-heads').value = config.model.n_heads;
    document.getElementById('n-heads-value').textContent = config.model.n_heads;
    
    document.getElementById('n-embed').value = config.model.n_embed;
    document.getElementById('n-embed-value').textContent = config.model.n_embed;
    
    document.getElementById('learning-rate').value = config.training.learning_rate;
    document.getElementById('batch-size').value = config.training.batch_size;
    document.getElementById('epochs').value = config.training.epochs;
    if (config.training.train_steps) {
        document.getElementById('train-steps').value = config.training.train_steps;
    }
    document.getElementById('total-epochs').textContent = config.training.epochs;
    
    // LR scheduler settings
    if (config.training.scheduler_type) {
        document.getElementById('scheduler-type').value = config.training.scheduler_type;
    }
    if (config.training.warmup_ratio !== undefined) {
        const warmupPercent = Math.round(config.training.warmup_ratio * 100);
        document.getElementById('warmup-ratio').value = warmupPercent;
        document.getElementById('warmup-ratio-value').textContent = warmupPercent;
    }
    if (config.training.min_lr_ratio !== undefined) {
        const minLrPercent = Math.round(config.training.min_lr_ratio * 100);
        document.getElementById('min-lr-ratio').value = minLrPercent;
        document.getElementById('min-lr-ratio-value').textContent = minLrPercent;
    }
}

// Initialize LR schedule chart
function initLRScheduleChart() {
    const ctx = document.getElementById('lr-schedule-chart').getContext('2d');
    lrScheduleChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Learning Rate',
                data: [],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                tension: 0.1,
                pointRadius: 0
            }, {
                label: 'Current Position',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.5)',
                pointRadius: 5,
                pointHoverRadius: 7,
                showLine: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Learning Rate',
                        color: '#e0e0e0'
                    },
                    grid: {
                        color: '#444'
                    },
                    ticks: {
                        color: '#e0e0e0',
                        callback: function(value) {
                            return value.toExponential(2);
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Training Step',
                        color: '#e0e0e0'
                    },
                    grid: {
                        color: '#444'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                return `LR: ${context.parsed.y.toExponential(4)}`;
                            }
                            return `Current: Step ${context.parsed.x}`;
                        }
                    }
                }
            }
        }
    });
}

// Calculate LR schedule client-side for preview
function calculateLRSchedule(config, numPoints = 200) {
    const maxLR = config.learning_rate;
    const totalSteps = config.train_steps || (config.batch_size * config.epochs);
    const warmupSteps = Math.floor(totalSteps * config.warmup_ratio);
    const tailStart = Math.floor(totalSteps * (1 - 0.2)); // 20% tail
    const minLR = maxLR * config.min_lr_ratio;
    
    const steps = [];
    const lrs = [];
    
    for (let i = 0; i <= numPoints; i++) {
        const step = Math.floor((i / numPoints) * totalSteps);
        steps.push(step);
        
        let lr;
        if (config.scheduler_type === 'warmup_cosine') {
            if (step < warmupSteps) {
                lr = maxLR * step / warmupSteps;
            } else if (step < tailStart) {
                const progress = (step - warmupSteps) / (tailStart - warmupSteps);
                const tailStartLR = maxLR * 0.3;
                lr = tailStartLR + (maxLR - tailStartLR) * 0.5 * (1 + Math.cos(Math.PI * progress));
            } else {
                const tailProgress = (step - tailStart) / (totalSteps - tailStart);
                const tailStartLR = maxLR * 0.3;
                lr = minLR + (tailStartLR - minLR) * 0.5 * (1 + Math.cos(Math.PI * tailProgress));
            }
        } else if (config.scheduler_type === 'warmup_linear') {
            if (step < warmupSteps) {
                lr = maxLR * step / warmupSteps;
            } else if (step < tailStart) {
                const progress = (step - warmupSteps) / (tailStart - warmupSteps);
                const tailStartLR = maxLR * 0.3;
                lr = maxLR - (maxLR - tailStartLR) * progress;
            } else {
                const tailProgress = (step - tailStart) / (totalSteps - tailStart);
                const tailStartLR = maxLR * 0.3;
                lr = tailStartLR - (tailStartLR - minLR) * tailProgress;
            }
        } else if (config.scheduler_type === 'warmup_constant') {
            if (step < warmupSteps) {
                lr = maxLR * step / warmupSteps;
            } else if (step < tailStart) {
                lr = maxLR;
            } else {
                const tailProgress = (step - tailStart) / (totalSteps - tailStart);
                lr = minLR + (maxLR - minLR) * 0.5 * (1 + Math.cos(Math.PI * tailProgress));
            }
        } else { // onecycle
            const pctStart = 0.3;
            const divFactor = 25.0;
            const finalDivFactor = 10000.0;
            const initialLR = maxLR / divFactor;
            const minLROneCycle = maxLR / finalDivFactor;
            const warmupStepsOne = Math.floor(totalSteps * pctStart);
            
            if (step < warmupStepsOne) {
                const pct = step / warmupStepsOne;
                lr = initialLR + (maxLR - initialLR) * pct;
            } else {
                const stepNum = step - warmupStepsOne;
                const stepMax = totalSteps - warmupStepsOne;
                const pct = stepNum / stepMax;
                lr = minLROneCycle + (maxLR - minLROneCycle) * 0.5 * (1 + Math.cos(Math.PI * pct));
            }
        }
        
        lrs.push(lr);
    }
    
    return { steps, lrs };
}

// Load and display LR schedule
async function loadLRSchedule() {
    try {
        const response = await fetch('/api/lr_schedule');
        const data = await response.json();
        
        if (data.steps && data.learning_rates) {
            // Update chart
            lrScheduleChart.data.labels = data.steps;
            lrScheduleChart.data.datasets[0].data = data.learning_rates;
            lrScheduleChart.update();
            
            // Update info
            const scheduleInfo = data.schedule_info;
            if (scheduleInfo) {
                document.getElementById('lr-schedule-type').textContent = 
                    scheduleInfo.type.replace(/([A-Z])/g, ' $1').trim();
                updateLRPhase(scheduleInfo);
            }
        }
    } catch (error) {
        console.error('Failed to load LR schedule:', error);
    }
}

// Update LR schedule preview (client-side)
function updateLRPreview() {
    if (!lrScheduleChart) return;
    
    // Get current values from UI
    const trainStepsValue = document.getElementById('train-steps').value;
    const config = {
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        batch_size: parseInt(document.getElementById('batch-size').value),
        epochs: parseInt(document.getElementById('epochs').value),
        train_steps: trainStepsValue ? parseInt(trainStepsValue) : null,
        scheduler_type: document.getElementById('scheduler-type').value,
        warmup_ratio: parseInt(document.getElementById('warmup-ratio').value) / 100,
        min_lr_ratio: parseInt(document.getElementById('min-lr-ratio').value) / 100
    };
    
    // Calculate schedule
    const { steps, lrs } = calculateLRSchedule(config);
    
    // Update chart
    lrScheduleChart.data.labels = steps;
    lrScheduleChart.data.datasets[0].data = lrs;
    lrScheduleChart.update('none'); // No animation for smooth updates
}

// Update current position on LR schedule
function updateLRSchedulePosition(currentStep) {
    if (lrScheduleChart && currentStep !== undefined) {
        const dataset = lrScheduleChart.data.datasets[1];
        const labels = lrScheduleChart.data.labels;
        
        // Find the closest step in the schedule
        const closestIndex = labels.reduce((prev, curr, index) => {
            return Math.abs(curr - currentStep) < Math.abs(labels[prev] - currentStep) ? index : prev;
        }, 0);
        
        // Update current position marker
        dataset.data = labels.map((step, index) => {
            return index === closestIndex ? lrScheduleChart.data.datasets[0].data[index] : null;
        });
        
        lrScheduleChart.update('none');
    }
}

// Update LR phase display
function updateLRPhase(scheduleInfo) {
    const currentStep = scheduleInfo.current_step;
    const warmupSteps = scheduleInfo.warmup_steps;
    const tailStart = scheduleInfo.tail_start;
    
    let phase = 'Not Started';
    if (currentStep < warmupSteps) {
        phase = `Warmup (${((currentStep / warmupSteps) * 100).toFixed(1)}%)`;
    } else if (currentStep < tailStart) {
        phase = 'Main Training';
    } else {
        phase = 'Fine Polish (Tail 20%)';
    }
    
    document.getElementById('lr-phase').textContent = phase;
}

// Initialize architecture visualization
async function initArchitectureVisualization() {
    try {
        architectureViz = new ArchitectureVisualizer('architecture-container');
        
        // Load initial architecture
        const response = await fetch('/api/architecture');
        const architecture = await response.json();
        architectureViz.updateArchitecture(architecture);
        
        // Handle component selection
        document.getElementById('architecture-container').addEventListener('component-selected', (event) => {
            showComponentDetails(event.detail);
        });
        
    } catch (error) {
        console.error('Failed to initialize architecture visualization:', error);
    }
}

// Update activations in visualization
function updateActivations(activationStates) {
    if (architectureViz && animationsEnabled) {
        architectureViz.updateStates(activationStates);
        
        // In visualization mode, find the most active component
        if (currentConfig.training.visualization_mode) {
            let activeComponent = null;
            let activePhase = 'Idle';
            
            for (const [id, state] of Object.entries(activationStates)) {
                if (state === 'forward') {
                    activeComponent = id;
                    activePhase = 'Forward Pass';
                    break;
                } else if (state === 'backward') {
                    activeComponent = id;
                    activePhase = 'Backward Pass';
                }
            }
            
            if (activeComponent) {
                architectureViz.updateTrainingStatus(activePhase, null, activeComponent);
            }
        }
    }
}

// Show component details
function showComponentDetails(details) {
    const detailsDiv = document.getElementById('component-details');
    const nameEl = document.getElementById('component-name');
    const paramsEl = document.getElementById('component-params');
    
    nameEl.textContent = details.name;
    paramsEl.innerHTML = '';
    
    // Add type
    paramsEl.innerHTML += `<div class="param"><span class="param-name">Type:</span><span class="param-value">${details.type}</span></div>`;
    
    // Add state
    paramsEl.innerHTML += `<div class="param"><span class="param-name">State:</span><span class="param-value">${details.state}</span></div>`;
    
    // Add other parameters
    if (details.params) {
        Object.entries(details.params).forEach(([key, value]) => {
            paramsEl.innerHTML += `<div class="param"><span class="param-name">${key}:</span><span class="param-value">${value}</span></div>`;
        });
    }
    
    detailsDiv.classList.add('visible');
    
    // Hide after 5 seconds
    setTimeout(() => {
        detailsDiv.classList.remove('visible');
    }, 5000);
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    initChart();
    initLRScheduleChart();
    loadConfig();
    initArchitectureVisualization();
    
    // Load metrics history
    loadMetricsHistory();
    
    // Load LR schedule
    loadLRSchedule();
    
    // Get initial status
    getTrainingStatus();
    
    // Poll for status every 2 seconds
    setInterval(getTrainingStatus, 2000);
    
    // Training controls
    document.getElementById('start-training').addEventListener('click', async () => {
        try {
            // Reset the chart for new training session
            resetChart();
            
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'start' })
            });
            const result = await response.json();
            console.log('Training:', result.status);
        } catch (error) {
            console.error('Failed to start training:', error);
        }
    });
    
    document.getElementById('stop-training').addEventListener('click', async () => {
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'stop' })
            });
            const result = await response.json();
            console.log('Training:', result.status);
        } catch (error) {
            console.error('Failed to stop training:', error);
        }
    });
    
    // Architecture configuration
    document.getElementById('apply-architecture').addEventListener('click', async () => {
        const modelConfig = {
            ...currentConfig.model,
            use_layer_norm: document.getElementById('use-layer-norm').checked,
            use_residual: document.getElementById('use-residual').checked,
            norm_position: document.querySelector('input[name="norm-position"]:checked').value,
            n_layers: parseInt(document.getElementById('n-layers').value),
            n_heads: parseInt(document.getElementById('n-heads').value),
            n_embed: parseInt(document.getElementById('n-embed').value)
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelConfig })
            });
            const result = await response.json();
            currentConfig = result.config;
            
            // Reload architecture visualization
            const archResponse = await fetch('/api/architecture');
            const architecture = await archResponse.json();
            architectureViz.updateArchitecture(architecture);
            
            alert('Architecture updated successfully!');
        } catch (error) {
            console.error('Failed to update architecture:', error);
            alert('Failed to update architecture');
        }
    });
    
    // Training parameters
    document.getElementById('update-params').addEventListener('click', async () => {
        const trainStepsValue = document.getElementById('train-steps').value;
        const trainingConfig = {
            ...currentConfig.training,
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            batch_size: parseInt(document.getElementById('batch-size').value),
            epochs: parseInt(document.getElementById('epochs').value),
            train_steps: trainStepsValue ? parseInt(trainStepsValue) : null,
            scheduler_type: document.getElementById('scheduler-type').value,
            warmup_ratio: parseInt(document.getElementById('warmup-ratio').value) / 100,
            min_lr_ratio: parseInt(document.getElementById('min-lr-ratio').value) / 100
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ training: trainingConfig })
            });
            const result = await response.json();
            currentConfig = result.config;
            document.getElementById('total-steps').textContent = trainingConfig.train_steps;
            alert('Parameters updated successfully!');
        } catch (error) {
            console.error('Failed to update parameters:', error);
            alert('Failed to update parameters');
        }
    });
    
    // Text generation
    document.getElementById('generate-text').addEventListener('click', async () => {
        const prompt = document.getElementById('prompt-input').value;
        const maxTokens = parseInt(document.getElementById('max-tokens').value);
        const temperature = parseFloat(document.getElementById('temperature').value);
        
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    max_tokens: maxTokens,
                    temperature: temperature
                })
            });
            const result = await response.json();
            document.getElementById('generated-text').textContent = result.generated;
        } catch (error) {
            console.error('Failed to generate text:', error);
            document.getElementById('generated-text').textContent = 'Error generating text';
        }
    });
    
    // Sliders
    document.getElementById('n-layers').addEventListener('input', (e) => {
        document.getElementById('n-layers-value').textContent = e.target.value;
    });
    
    document.getElementById('n-heads').addEventListener('input', (e) => {
        document.getElementById('n-heads-value').textContent = e.target.value;
    });
    
    document.getElementById('n-embed').addEventListener('input', (e) => {
        document.getElementById('n-embed-value').textContent = e.target.value;
    });
    
    document.getElementById('max-tokens').addEventListener('input', (e) => {
        document.getElementById('max-tokens-value').textContent = e.target.value;
    });
    
    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('temperature-value').textContent = e.target.value;
    });
    
    // LR Schedule controls with live preview
    document.getElementById('warmup-ratio').addEventListener('input', (e) => {
        document.getElementById('warmup-ratio-value').textContent = e.target.value;
        updateLRPreview(); // Live preview
    });
    
    document.getElementById('min-lr-ratio').addEventListener('input', (e) => {
        document.getElementById('min-lr-ratio-value').textContent = e.target.value;
        updateLRPreview(); // Live preview
    });
    
    // Also update preview when other relevant parameters change
    document.getElementById('scheduler-type').addEventListener('change', updateLRPreview);
    document.getElementById('learning-rate').addEventListener('input', updateLRPreview);
    document.getElementById('epochs').addEventListener('input', updateLRPreview);
    document.getElementById('batch-size').addEventListener('input', updateLRPreview);
    document.getElementById('train-steps').addEventListener('input', updateLRPreview);
    
    document.getElementById('preview-schedule').addEventListener('click', async () => {
        // Get current values from UI
        const trainStepsValue = document.getElementById('train-steps').value;
        const trainingConfig = {
            ...currentConfig.training,
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            batch_size: parseInt(document.getElementById('batch-size').value),
            epochs: parseInt(document.getElementById('epochs').value),
            train_steps: trainStepsValue ? parseInt(trainStepsValue) : null,
            scheduler_type: document.getElementById('scheduler-type').value,
            warmup_ratio: parseInt(document.getElementById('warmup-ratio').value) / 100,
            min_lr_ratio: parseInt(document.getElementById('min-lr-ratio').value) / 100
        };
        
        try {
            // Update config and wait for response
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ training: trainingConfig })
            });
            const result = await response.json();
            currentConfig = result.config;
            
            // Now load the updated schedule
            await loadLRSchedule();
        } catch (error) {
            console.error('Failed to preview schedule:', error);
            alert('Failed to preview schedule');
        }
    });
    
    // Architecture controls
    document.getElementById('reset-zoom').addEventListener('click', () => {
        if (architectureViz) {
            architectureViz.svg.transition()
                .duration(750)
                .call(
                    d3.zoom().transform,
                    d3.zoomIdentity
                );
        }
    });
    
    document.getElementById('toggle-animations').addEventListener('click', (e) => {
        animationsEnabled = !animationsEnabled;
        e.target.textContent = animationsEnabled ? 'Disable Animations' : 'Enable Animations';
        e.target.classList.toggle('active', animationsEnabled);
    });
    
    // Visualization mode controls
    const visualizationModeCheckbox = document.getElementById('visualization-mode');
    const visualizationSpeedControl = document.getElementById('visualization-speed-control');
    const visualizationSpeedSlider = document.getElementById('visualization-speed');
    const visualizationSpeedValue = document.getElementById('visualization-speed-value');
    
    visualizationModeCheckbox.addEventListener('change', async (e) => {
        const enabled = e.target.checked;
        visualizationSpeedControl.style.display = enabled ? 'flex' : 'none';
        
        // Update training config
        const trainingConfig = {
            ...currentConfig.training,
            visualization_mode: enabled,
            visualization_speed_ratio: parseInt(visualizationSpeedSlider.value) / 100
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ training: trainingConfig })
            });
            const result = await response.json();
            currentConfig = result.config;
            
            if (enabled) {
                alert('Visualization mode enabled. Training will run at ' + visualizationSpeedSlider.value + '% speed.');
                if (architectureViz) {
                    architectureViz.showVisualizationStatus(true);
                }
            } else {
                if (architectureViz) {
                    architectureViz.showVisualizationStatus(false);
                }
            }
        } catch (error) {
            console.error('Failed to update visualization mode:', error);
        }
    });
    
    visualizationSpeedSlider.addEventListener('input', async (e) => {
        const speedPercent = e.target.value;
        visualizationSpeedValue.textContent = speedPercent;
        
        // Update training config
        const trainingConfig = {
            ...currentConfig.training,
            visualization_mode: visualizationModeCheckbox.checked,
            visualization_speed_ratio: parseInt(speedPercent) / 100
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ training: trainingConfig })
            });
            const result = await response.json();
            currentConfig = result.config;
        } catch (error) {
            console.error('Failed to update visualization speed:', error);
        }
    });
});

// Handle training completion
function handleTrainingComplete(data) {
    console.log('Training completed!', data);
    
    // Update final metrics display
    const metrics = data.final_metrics;
    document.getElementById('current-loss').textContent = metrics.train_loss.toFixed(4);
    document.getElementById('current-perplexity').textContent = metrics.perplexity.toFixed(2);
    
    // Show completion message
    const completionDiv = document.getElementById('training-completion');
    if (completionDiv) {
        completionDiv.innerHTML = `
            <h3>Training Complete!</h3>
            <p>${data.message}</p>
            <p>Final Perplexity: ${metrics.perplexity.toFixed(2)}</p>
            <p>Final Val Perplexity: ${metrics.val_perplexity.toFixed(2)}</p>
        `;
        completionDiv.style.display = 'block';
    }
    
    // Update status
    getTrainingStatus();
}

// Get training status
async function getTrainingStatus() {
    try {
        const response = await fetch('/api/train/status');
        const status = await response.json();
        updateStatus(status);
    } catch (error) {
        console.error('Failed to get training status:', error);
    }
}

// Load metrics history
async function loadMetricsHistory() {
    try {
        const response = await fetch('/api/metrics/history');
        const data = await response.json();
        
        if (data.metrics && data.metrics.length > 0) {
            // Populate chart with historical data
            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.data.datasets[1].data = [];
            
            data.metrics.forEach(metric => {
                if (metric.step % 100 === 0) {
                    lossChart.data.labels.push(metric.step);
                    lossChart.data.datasets[0].data.push(metric.train_loss);
                    if (metric.val_loss > 0) {
                        lossChart.data.datasets[1].data.push(metric.val_loss);
                    }
                }
            });
            
            lossChart.update('none');
        }
    } catch (error) {
        console.error('Failed to load metrics history:', error);
    }
}