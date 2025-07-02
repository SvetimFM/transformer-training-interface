let ws = null;
let lossChart = null;
let lrScheduleChart = null;
let currentConfig = {};
let architectureViz = null;
let animationsEnabled = true;

// Tab switching functionality
function initTabSwitching() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const dropdownButton = document.querySelector('.dropdown-button');
    const dropdownContent = document.querySelector('.dropdown-content');
    
    // Handle dropdown toggle
    if (dropdownButton) {
        dropdownButton.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdownContent.classList.toggle('show');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            dropdownContent.classList.remove('show');
        });
        
        // Prevent dropdown from closing when clicking inside
        dropdownContent.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Update active states
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Only mark main transformer tab as active in the main nav
            if (targetTab === 'transformer') {
                button.classList.add('active');
            }
            
            document.getElementById(`${targetTab}-tab`).classList.add('active');
            
            // Close dropdown after selection
            if (dropdownContent) {
                dropdownContent.classList.remove('show');
            }
            
            // Initialize tab-specific content if needed
            if (targetTab === 'pcn-experiments') {
                initPCNExperiments();
            } else if (targetTab === 'hybrid-models') {
                initHybridModels();
            }
        });
    });
}

// Initialize PCN Experiments tab
function initPCNExperiments() {
    // Initialize PCN charts if they haven't been created yet
    if (typeof initPCNCharts === 'function') {
        initPCNCharts();
    }
}

// Initialize Hybrid Models tab
function initHybridModels() {
    // Initialize hybrid architecture visualization
    if (typeof initHybridArchitecture === 'function') {
        initHybridArchitecture();
    }
    
    // Update architecture description on selection change
    const archSelector = document.getElementById('hybrid-architecture');
    if (archSelector && !archSelector.hasListener) {
        archSelector.hasListener = true;
        archSelector.addEventListener('change', updateArchitectureDescription);
    }
}

// Update architecture description based on selection
function updateArchitectureDescription() {
    const archSelector = document.getElementById('hybrid-architecture');
    const descriptionDiv = document.getElementById('arch-description');
    
    const descriptions = {
        'pcn-ff': 'PCN replaces the feedforward networks in transformer blocks, bringing biological plausibility to the computation-heavy MLP layers.',
        'alternating': 'Alternates between attention and PCN layers, allowing each mechanism to specialize in different aspects of sequence processing.',
        'hierarchical': 'PCN processes features at a lower level, then passes refined representations to the transformer for sequence modeling.',
        'dual-stream': 'Runs PCN and transformer in parallel streams, combining their outputs for richer representations.',
        'pcn-positional': 'Uses PCN to learn adaptive positional encodings that adjust based on the input context.'
    };
    
    descriptionDiv.innerHTML = `<p>${descriptions[archSelector.value] || 'Select an architecture to see its description.'}</p>`;
}

// Update layer configuration based on number of layers
function updateLayerConfig(numLayers) {
    const simpleConfig = document.getElementById('simple-config');
    const perLayerConfig = document.getElementById('per-layer-config');
    const layerConfigs = document.getElementById('layer-configs');
    
    if (numLayers === 1) {
        simpleConfig.style.display = 'block';
        perLayerConfig.style.display = 'none';
    } else {
        simpleConfig.style.display = 'none';
        perLayerConfig.style.display = 'block';
        
        // Generate per-layer configuration
        layerConfigs.innerHTML = '';
        
        // Get current embed dimension for head validation
        const embedDim = parseInt(document.getElementById('n-embed').value);
        const maxHeads = Math.floor(embedDim / 4); // Minimum head size of 4
        
        for (let i = 0; i < numLayers; i++) {
            const layerDiv = document.createElement('div');
            layerDiv.className = 'layer-config-item';
            layerDiv.innerHTML = `
                <h4>Layer ${i + 1}</h4>
                <div class="layer-config-grid">
                    <div>
                        <label>Heads: <span id="layer-${i}-heads-value">8</span></label>
                        <input type="range" id="layer-${i}-heads" min="1" max="${Math.min(maxHeads, 16)}" value="${Math.min(8, maxHeads)}" 
                               class="slider layer-heads" data-layer="${i}">
                    </div>
                    <div>
                        <label>Hidden: <span id="layer-${i}-hidden-value">4</span>x</label>
                        <input type="range" id="layer-${i}-hidden" min="1" max="8" value="4" 
                               class="slider layer-hidden" data-layer="${i}">
                    </div>
                </div>
            `;
            layerConfigs.appendChild(layerDiv);
        }
        
        // Add event listeners to new sliders
        document.querySelectorAll('.layer-heads').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const layer = e.target.dataset.layer;
                document.getElementById(`layer-${layer}-heads-value`).textContent = e.target.value;
                calculateTotalParams();
            });
        });
        
        document.querySelectorAll('.layer-hidden').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const layer = e.target.dataset.layer;
                document.getElementById(`layer-${layer}-hidden-value`).textContent = e.target.value;
                calculateTotalParams();
            });
        });
    }
}

// Calculate total parameters
function calculateTotalParams() {
    try {
        // Avoid race condition - ensure config is loaded
        if (!currentConfig.model) return;
        
        const vocabSize = currentConfig.model?.vocab_size || 50000;
        const embedDim = parseInt(document.getElementById('n-embed').value);
        const numLayers = parseInt(document.getElementById('n-layers').value);
        
        let totalParams = 0;
        
        // Embedding parameters
        totalParams += vocabSize * embedDim; // Token embeddings
        totalParams += (currentConfig.model?.block_size || 128) * embedDim; // Position embeddings
        
        // Layer parameters
        if (numLayers === 1) {
            const hiddenMult = parseInt(document.getElementById('hidden-mult').value);
            
            // Attention parameters (Q, K, V, O projections)
            totalParams += 4 * embedDim * embedDim;
            
            // FFN parameters (Note: backend currently hardcoded to 4x)
            // TODO: Update backend FeedForward to accept hiddenMult parameter
            totalParams += embedDim * (4 * embedDim); // First linear (using 4x as per backend)
            totalParams += (4 * embedDim) * embedDim; // Second linear
            
            // Layer norm parameters (2 per layer)
            totalParams += 2 * embedDim;
        } else {
            for (let i = 0; i < numLayers; i++) {
                const hiddenMult = parseInt(document.getElementById(`layer-${i}-hidden`)?.value || 4);
                
                // Attention parameters
                totalParams += 4 * embedDim * embedDim;
                
                // FFN parameters (Note: backend currently hardcoded to 4x)
                totalParams += embedDim * (4 * embedDim);
                totalParams += (4 * embedDim) * embedDim;
                
                // Layer norm parameters
                totalParams += 2 * embedDim;
            }
        }
        
        // Final layer norm
        totalParams += embedDim; // Final layer norm
        
        // Output layers
        const numOutputLayers = parseInt(document.getElementById('n-output-layers').value);
        const outputHiddenDim = parseInt(document.getElementById('output-hidden-dim').value);
        
        if (numOutputLayers > 0) {
            // First output layer: embedDim -> outputHiddenDim
            totalParams += embedDim * outputHiddenDim;
            
            // Additional hidden layers: outputHiddenDim -> outputHiddenDim
            for (let i = 1; i < numOutputLayers; i++) {
                totalParams += outputHiddenDim * outputHiddenDim;
            }
            
            // Final projection: outputHiddenDim -> vocabSize
            totalParams += outputHiddenDim * vocabSize;
        } else {
            // Direct projection: embedDim -> vocabSize
            totalParams += embedDim * vocabSize;
        }
        
        // Format the number with commas
        const formatted = totalParams.toLocaleString();
        document.getElementById('total-params').textContent = formatted;
    } catch (error) {
        console.error('Error calculating parameters:', error);
        const paramsElement = document.getElementById('total-params');
        if (paramsElement) {
            paramsElement.textContent = 'Error';
        }
    }
}

// PCN WebSocket handlers
function updatePCNMetrics(data) {
    if (data.experiment === 'data_leakage') {
        // Update accuracy displays (with null checks)
        const accuracyLeaked = document.getElementById('pcn-accuracy-leaked');
        const accuracyClean = document.getElementById('pcn-accuracy-clean');
        const stepsLeaked = document.getElementById('pcn-steps-leaked');
        const stepsClean = document.getElementById('pcn-steps-clean');
        
        if (accuracyLeaked) accuracyLeaked.textContent = data.accuracy_claimed.toFixed(2) + '%';
        if (accuracyClean) accuracyClean.textContent = data.accuracy_realistic.toFixed(2) + '%';
        if (stepsLeaked) stepsLeaked.textContent = data.inference_steps;
        if (stepsClean) stepsClean.textContent = data.inference_steps;
        
        // Update comparison chart
        if (typeof updatePCNComparison === 'function') {
            updatePCNComparison(data.accuracy_claimed, data.accuracy_realistic, data.epoch);
        }
        
        // Update experiment outputs
        const leakedOutput = document.getElementById('pcn-output-leaked');
        const cleanOutput = document.getElementById('pcn-output-clean');
        
        if (leakedOutput) {
            leakedOutput.innerHTML = `
                <div class="output-metric"><strong>Epoch:</strong> ${data.epoch + 1}</div>
                <div class="output-metric"><strong>Accuracy:</strong> ${data.accuracy_claimed.toFixed(2)}%</div>
                <div class="output-metric"><strong>Method:</strong> test_generative(x, y)</div>
                <div class="output-metric"><strong>Issue:</strong> Labels used during inference</div>
                <div class="output-metric"><strong>Result:</strong> Artificially high accuracy</div>
            `;
        }
        
        if (cleanOutput) {
            cleanOutput.innerHTML = `
                <div class="output-metric"><strong>Epoch:</strong> ${data.epoch + 1}</div>
                <div class="output-metric"><strong>Accuracy:</strong> ${data.accuracy_realistic.toFixed(2)}%</div>
                <div class="output-metric"><strong>Method:</strong> test_discriminative(x)</div>
                <div class="output-metric"><strong>Issue:</strong> None - proper evaluation</div>
                <div class="output-metric"><strong>Result:</strong> Realistic accuracy for CIFAR-10</div>
            `;
        }
        
        // Update findings
        const findingsList = document.getElementById('pcn-findings-list');
        if (findingsList && data.accuracy_claimed > 90 && data.accuracy_realistic < 60) {
            findingsList.innerHTML = `
                <li>With label leakage: ${data.accuracy_claimed.toFixed(1)}% accuracy (matching paper claims)</li>
                <li>Without label leakage: ${data.accuracy_realistic.toFixed(1)}% accuracy (actual performance)</li>
                <li>Performance gap: ${(data.accuracy_claimed - data.accuracy_realistic).toFixed(1)}% difference</li>
                <li>Conclusion: PCN's claimed performance relies on unrealistic test conditions</li>
            `;
        }
    }
}

function updatePCNExploration(data) {
    // Update energy chart with both datasets
    if (typeof updatePCNEnergy === 'function' && data.energy_leaked && data.energy_clean) {
        updatePCNEnergy(data.energy_leaked, data.energy_clean);
    }
    
    // Update diversity chart with both datasets
    if (typeof updatePCNDiversity === 'function' && data.diversity_leaked && data.diversity_clean) {
        updatePCNDiversity(data.diversity_leaked, data.diversity_clean);
    }
}

function updateHybridMetrics(data) {
    // Update loss displays
    document.getElementById('hybrid-loss').textContent = data.hybrid_loss.toFixed(4);
    document.getElementById('hybrid-perplexity').textContent = data.hybrid_perplexity.toFixed(2);
    document.getElementById('baseline-loss').textContent = data.baseline_loss.toFixed(4);
    document.getElementById('baseline-perplexity').textContent = data.baseline_perplexity.toFixed(2);
    
    // Update performance chart
    if (typeof updateHybridPerformance === 'function') {
        updateHybridPerformance(data.hybrid_loss, data.baseline_loss);
    }
    
    // Bio-plausibility removed - not meaningful for this project
}

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
            } else if (data.type === 'pcn_metrics') {
                updatePCNMetrics(data.data);
            } else if (data.type === 'pcn_exploration') {
                updatePCNExploration(data.data);
            } else if (data.type === 'hybrid_metrics') {
                updateHybridMetrics(data.data);
            } else if (data.type === 'multi_lora_metrics') {
                updateMultiLoRAMetrics(data.data);
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
                tension: 0.1,
                spanGaps: true  // This will connect points across null values
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
    
    // Update performance metrics
    if ('tokens_per_second' in metrics) {
        document.getElementById('tokens-per-second').textContent = Math.round(metrics.tokens_per_second);
    }
    if ('gpu_memory_mb' in metrics) {
        document.getElementById('gpu-memory').textContent = metrics.gpu_memory_mb.toFixed(0);
    }
    if ('gradient_norm' in metrics && metrics.gradient_norm !== null) {
        document.getElementById('gradient-norm').textContent = metrics.gradient_norm.toFixed(3);
    }
    
    // Update architecture visualization status in visualization mode
    if (currentConfig.training.visualization_mode && architectureViz) {
        architectureViz.updateTrainingStatus('Training', metrics.step);
    }
    
    // Update chart - show every 50 steps instead of 100 for more responsive updates
    if (lossChart && metrics.step % 50 === 0) {
        // Always update training loss
        lossChart.data.labels.push(metrics.step);
        lossChart.data.datasets[0].data.push(metrics.train_loss);
        
        // Only add validation loss if it exists
        if ('val_loss' in metrics && metrics.val_loss !== null) {
            lossChart.data.datasets[1].data.push(metrics.val_loss);
        }
        
        // Keep only last 50 points
        if (lossChart.data.labels.length > 50) {
            lossChart.data.labels.shift();
            lossChart.data.datasets[0].data.shift();
            // Only shift validation data if it has the same length
            if (lossChart.data.datasets[1].data.length > 50) {
                lossChart.data.datasets[1].data.shift();
            }
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
    
    // Update button visibility based on training status
    const startButton = document.getElementById('start-training');
    const stopButton = document.getElementById('stop-training');
    if (startButton && stopButton) {
        if (status.is_training) {
            startButton.style.display = 'none';
            stopButton.style.display = 'inline-block';
        } else {
            startButton.style.display = 'inline-block';
            stopButton.style.display = 'none';
        }
    }
    
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
        // Calculate total params after config is loaded
        calculateTotalParams();
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
    
    // Output layer configuration
    if (config.model.output_activation) {
        document.getElementById('output-activation').value = config.model.output_activation;
    }
    if (config.model.n_output_layers !== undefined) {
        document.getElementById('n-output-layers').value = config.model.n_output_layers;
        document.getElementById('n-output-layers-value').textContent = config.model.n_output_layers;
        document.getElementById('output-hidden-config').style.display = 
            config.model.n_output_layers > 0 ? 'block' : 'none';
    }
    if (config.model.output_hidden_dim !== undefined) {
        document.getElementById('output-hidden-dim').value = config.model.output_hidden_dim;
        document.getElementById('output-hidden-dim-value').textContent = config.model.output_hidden_dim;
    }
    
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
    
    // Optimization settings
    if (config.training.compile_model !== undefined) {
        document.getElementById('compile-model').checked = config.training.compile_model;
    }
    if (config.training.use_amp !== undefined) {
        document.getElementById('use-amp').checked = config.training.use_amp;
    }
    if (config.training.gradient_accumulation_steps !== undefined) {
        document.getElementById('gradient-accumulation-steps').value = config.training.gradient_accumulation_steps;
    }
    if (config.training.gradient_clip_norm !== undefined) {
        document.getElementById('gradient-clip-norm').value = config.training.gradient_clip_norm;
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
    
    // Initialize tab switching
    initTabSwitching();
    
    // Initialize PCN charts on page load
    if (typeof initPCNCharts === 'function') {
        initPCNCharts();
    }
    
    // Initialize Multi-LoRA charts
    if (typeof initMultiLoRACharts === 'function') {
        initMultiLoRACharts();
        initMultiLoRAConfig();
        createTokenHeatmap('token-selection-heatmap');
    }
    
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
            
            // Force a status update after stopping
            setTimeout(getTrainingStatus, 500);
        } catch (error) {
            console.error('Failed to stop training:', error);
        }
    });
    
    // Architecture configuration
    document.getElementById('apply-architecture').addEventListener('click', async () => {
        const n_heads = parseInt(document.getElementById('n-heads').value);
        const n_embed = parseInt(document.getElementById('n-embed').value);
        
        // Validate configuration
        if (n_embed % n_heads !== 0) {
            const headSize = n_embed / n_heads;
            const validSizes = [];
            for (let i = 8; i <= 16; i++) {
                validSizes.push(n_heads * i);
            }
            alert(`Invalid configuration: n_embed (${n_embed}) must be divisible by n_heads (${n_heads}).\n` +
                  `Current head_size would be ${headSize.toFixed(2)}, but it must be an integer.\n\n` +
                  `Try n_embed values like: ${validSizes.join(', ')}`);
            return;
        }
        
        const headSize = n_embed / n_heads;
        if (headSize < 4) {
            alert(`Invalid configuration: head_size (${headSize}) is too small.\n` +
                  `With n_heads=${n_heads}, you need n_embed >= ${n_heads * 4}.\n` +
                  `Consider reducing n_heads or increasing n_embed.`);
            return;
        }
        
        const modelConfig = {
            ...currentConfig.model,
            use_layer_norm: document.getElementById('use-layer-norm').checked,
            use_residual: document.getElementById('use-residual').checked,
            norm_position: document.querySelector('input[name="norm-position"]:checked').value,
            n_layers: parseInt(document.getElementById('n-layers').value),
            n_heads: n_heads,
            n_embed: n_embed,
            output_activation: document.getElementById('output-activation').value,
            n_output_layers: parseInt(document.getElementById('n-output-layers').value),
            output_hidden_dim: parseInt(document.getElementById('output-hidden-dim').value)
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
            min_lr_ratio: parseInt(document.getElementById('min-lr-ratio').value) / 100,
            compile_model: document.getElementById('compile-model').checked,
            use_amp: document.getElementById('use-amp').checked,
            gradient_accumulation_steps: parseInt(document.getElementById('gradient-accumulation-steps').value),
            gradient_clip_norm: parseFloat(document.getElementById('gradient-clip-norm').value)
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
    
    // Function to validate and show head size
    function updateHeadSizeDisplay() {
        const n_heads = parseInt(document.getElementById('n-heads').value);
        const n_embed = parseInt(document.getElementById('n-embed').value);
        const headSizeDisplay = document.getElementById('head-size-display');
        
        if (!headSizeDisplay) {
            // Create head size display if it doesn't exist
            const embedLabel = document.querySelector('label[for="n-embed"]').parentElement;
            const display = document.createElement('div');
            display.id = 'head-size-display';
            display.style.marginTop = '10px';
            display.style.fontSize = '14px';
            embedLabel.appendChild(display);
        }
        
        const display = document.getElementById('head-size-display');
        
        if (n_embed % n_heads !== 0) {
            const headSize = n_embed / n_heads;
            display.innerHTML = `<span style="color: #ef4444;">⚠️ Invalid: head_size = ${headSize.toFixed(2)} (must be integer)</span>`;
            display.style.display = 'block';
        } else {
            const headSize = n_embed / n_heads;
            if (headSize < 4) {
                display.innerHTML = `<span style="color: #ef4444;">⚠️ head_size = ${headSize} (too small, need >= 4)</span>`;
            } else if ((headSize & (headSize - 1)) !== 0) {
                display.innerHTML = `<span style="color: #f59e0b;">ℹ️ head_size = ${headSize} (works, but not power of 2)</span>`;
            } else {
                display.innerHTML = `<span style="color: #4ade80;">✓ head_size = ${headSize}</span>`;
            }
            display.style.display = 'block';
        }
    }
    
    // Sliders
    document.getElementById('n-layers').addEventListener('input', (e) => {
        document.getElementById('n-layers-value').textContent = e.target.value;
        updateLayerConfig(parseInt(e.target.value));
        calculateTotalParams();
    });
    
    document.getElementById('n-heads').addEventListener('input', (e) => {
        document.getElementById('n-heads-value').textContent = e.target.value;
        updateHeadSizeDisplay();
        calculateTotalParams();
    });
    
    document.getElementById('n-embed').addEventListener('input', (e) => {
        document.getElementById('n-embed-value').textContent = e.target.value;
        updateHeadSizeDisplay();
        calculateTotalParams();
    });
    
    // Hidden layer multiplier
    document.getElementById('hidden-mult').addEventListener('input', (e) => {
        document.getElementById('hidden-mult-value').textContent = e.target.value;
        calculateTotalParams();
    });
    
    
    // Output layer controls
    document.getElementById('n-output-layers').addEventListener('input', (e) => {
        const numOutputLayers = parseInt(e.target.value);
        document.getElementById('n-output-layers-value').textContent = numOutputLayers;
        
        // Show/hide hidden dimension control
        const hiddenConfig = document.getElementById('output-hidden-config');
        hiddenConfig.style.display = numOutputLayers > 0 ? 'block' : 'none';
        
        calculateTotalParams();
    });
    
    document.getElementById('output-hidden-dim').addEventListener('input', (e) => {
        document.getElementById('output-hidden-dim-value').textContent = e.target.value;
        calculateTotalParams();
    });
    
    document.getElementById('output-activation').addEventListener('change', calculateTotalParams);
    
    // Initial validation display
    updateHeadSizeDisplay();
    // calculateTotalParams() is called after config loads to avoid race condition
    
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
    
    // Detail level toggle
    document.getElementById('detail-level').addEventListener('change', (e) => {
        if (architectureViz) {
            const detailLevel = e.target.checked ? 'high' : 'low';
            architectureViz.setDetailLevel(detailLevel);
        }
    });
    
    // Multi-LoRA training controls
    document.getElementById('start-multi-lora-training')?.addEventListener('click', async () => {
        const config = {
            num_loras: parseInt(document.getElementById('num-loras').value),
            rank: parseInt(document.getElementById('lora-rank').value),
            alpha: parseInt(document.getElementById('lora-alpha').value),
            selection_mode: document.querySelector('input[name="lora-selection-mode"]:checked').value,
            temperature: parseFloat(document.getElementById('selection-temperature').value),
            target_modules: Array.from(document.querySelectorAll('#multi-lora-config-panel .checkbox-group input:checked'))
                .map(cb => cb.value)
        };
        
        try {
            const response = await fetch('/api/multi-lora/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            const result = await response.json();
            console.log('Multi-LoRA training started:', result);
        } catch (error) {
            console.error('Failed to start Multi-LoRA training:', error);
        }
    });
    
    document.getElementById('stop-multi-lora-training')?.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/multi-lora/stop', {
                method: 'POST'
            });
            const result = await response.json();
            console.log('Multi-LoRA training stopped:', result);
        } catch (error) {
            console.error('Failed to stop Multi-LoRA training:', error);
        }
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
            
            // Separate arrays for training and validation
            const valLossData = [];
            const valLossLabels = [];
            
            data.metrics.forEach(metric => {
                if (metric.step % 100 === 0) {
                    lossChart.data.labels.push(metric.step);
                    lossChart.data.datasets[0].data.push(metric.train_loss);
                    
                    // Only add validation data points when they exist
                    if ('val_loss' in metric && metric.val_loss !== null && metric.val_loss !== undefined) {
                        valLossLabels.push(metric.step);
                        valLossData.push(metric.val_loss);
                    }
                }
            });
            
            // For validation loss, we need to align it with the correct steps
            lossChart.data.datasets[1].data = lossChart.data.labels.map(step => {
                const idx = valLossLabels.indexOf(step);
                return idx >= 0 ? valLossData[idx] : null;
            });
            
            lossChart.update('none');
        }
    } catch (error) {
        console.error('Failed to load metrics history:', error);
    }
}