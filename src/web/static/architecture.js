class ArchitectureVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = null;
        this.components = {};
        this.detailLevel = 'low'; // 'low' or 'high'
        this.layout = {
            width: 800,
            height: 600,
            nodeWidth: 120,
            nodeHeight: 40,
            headSize: 30,
            horizontalSpacing: 150,
            verticalSpacing: 80,
            padding: 40
        };
        this.animationDuration = 300;
        this.activeAnimations = new Set();
        
        this.init();
    }
    
    init() {
        // Create SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.layout.width} ${this.layout.height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');
        
        // Add defs for gradients and filters
        const defs = this.svg.append('defs');
        
        // Glow filter for active components
        const glowFilter = defs.append('filter')
            .attr('id', 'glow');
        glowFilter.append('feGaussianBlur')
            .attr('stdDeviation', '3')
            .attr('result', 'coloredBlur');
        const feMerge = glowFilter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
        
        // Create main group for zooming/panning
        this.mainGroup = this.svg.append('g')
            .attr('class', 'main-group');
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {
                this.mainGroup.attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Add status overlay for visualization mode
        this.statusOverlay = this.svg.append('g')
            .attr('class', 'status-overlay')
            .attr('transform', 'translate(10, 30)');
            
        this.statusText = this.statusOverlay.append('text')
            .attr('class', 'status-text')
            .attr('fill', '#4ade80')
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .style('display', 'none');
            
        this.stepText = this.statusOverlay.append('text')
            .attr('class', 'step-text')
            .attr('fill', '#f59e0b')
            .attr('font-size', '14px')
            .attr('y', 20)
            .style('display', 'none');
    }
    
    updateArchitecture(architectureData) {
        const { components, root_components } = architectureData;
        this.components = components;
        
        // Clear existing visualization
        this.mainGroup.selectAll('*').remove();
        
        // Calculate layout
        const positions = this.calculateLayout(components, root_components);
        
        // Draw connections first (so they appear behind nodes)
        this.drawConnections(components, positions);
        
        // Draw components (filter based on detail level)
        Object.entries(components).forEach(([id, component]) => {
            if (this.shouldShowComponent(component)) {
                this.drawComponent(id, component, positions[id]);
            }
        });
    }
    
    calculateLayout(components, rootIds) {
        const positions = {};
        let currentY = this.layout.padding;
        
        // Group components by layer
        const layers = this.groupByLayers(components, rootIds);
        
        // Track transformer block boundaries
        const transformerBlocks = {};
        
        layers.forEach((layer, layerIndex) => {
            let maxHeight = 0;
            const layerComponents = [];
            
            // First pass: determine sizes and filter visible components
            layer.forEach(componentId => {
                const component = components[componentId];
                
                // Skip components that shouldn't be shown
                if (!this.shouldShowComponent(component)) {
                    return;
                }
                
                let width = this.layout.nodeWidth;
                let height = this.layout.nodeHeight;
                
                // Adjust size based on component type
                if (component.type === 'attention_head') {
                    width = height = this.layout.headSize;
                } else if (['matmul', 'add', 'residual_add', 'concat', 'split', 'softmax'].includes(component.type)) {
                    width = height = 50; // Diamonds are smaller
                } else if (['activation', 'dropout'].includes(component.type)) {
                    width = height = 40; // Circles are smaller
                } else if (component.type === 'transformer_block') {
                    // Skip transformer blocks in positioning, but track them
                    transformerBlocks[componentId] = {
                        startY: currentY,
                        components: []
                    };
                    return;
                }
                
                layerComponents.push({
                    id: componentId,
                    component: component,
                    width: width,
                    height: height
                });
                
                maxHeight = Math.max(maxHeight, height);
            });
            
            // Second pass: position components
            const layerWidth = layerComponents.length * this.layout.horizontalSpacing;
            const startX = (this.layout.width - layerWidth) / 2;
            
            layerComponents.forEach((item, index) => {
                positions[item.id] = {
                    x: startX + index * this.layout.horizontalSpacing,
                    y: currentY,
                    width: item.width,
                    height: item.height
                };
                
                // Track which transformer block this component belongs to
                let parentBlock = this.findParentTransformerBlock(item.component, components);
                if (parentBlock && transformerBlocks[parentBlock]) {
                    transformerBlocks[parentBlock].components.push(item.id);
                }
            });
            
            if (layerComponents.length > 0) {
                currentY += maxHeight + this.layout.verticalSpacing;
            }
        });
        
        // Position transformer blocks around their children
        Object.entries(transformerBlocks).forEach(([blockId, blockInfo]) => {
            if (blockInfo.components.length > 0) {
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                
                blockInfo.components.forEach(compId => {
                    const pos = positions[compId];
                    if (pos) {
                        minX = Math.min(minX, pos.x);
                        maxX = Math.max(maxX, pos.x + pos.width);
                        minY = Math.min(minY, pos.y);
                        maxY = Math.max(maxY, pos.y + pos.height);
                    }
                });
                
                positions[blockId] = {
                    x: minX - 30,
                    y: minY - 30,
                    width: maxX - minX + 60,
                    height: maxY - minY + 60
                };
            }
        });
        
        // Update SVG height if needed
        const totalHeight = currentY + this.layout.padding;
        if (totalHeight > this.layout.height) {
            this.layout.height = totalHeight;
            this.svg.attr('viewBox', `0 0 ${this.layout.width} ${this.layout.height}`);
        }
        
        return positions;
    }
    
    findParentTransformerBlock(component, components) {
        let current = component;
        while (current && current.parent_id) {
            const parent = components[current.parent_id];
            if (parent && parent.type === 'transformer_block') {
                return current.parent_id;
            }
            current = parent;
        }
        return null;
    }
    
    groupByLayers(components, rootIds) {
        const layers = [];
        const visited = new Set();
        
        // BFS to group components by depth
        let currentLayer = rootIds;
        
        while (currentLayer.length > 0) {
            layers.push([...currentLayer]);
            const nextLayer = [];
            
            currentLayer.forEach(id => {
                visited.add(id);
                const component = components[id];
                if (component && component.children_ids) {
                    component.children_ids.forEach(childId => {
                        if (!visited.has(childId)) {
                            nextLayer.push(childId);
                        }
                    });
                }
            });
            
            currentLayer = nextLayer;
        }
        
        return layers;
    }
    
    drawConnections(components, positions) {
        const connectionsGroup = this.mainGroup.append('g')
            .attr('class', 'connections');
        
        // Helper function to find the next visible descendant
        const findNextVisibleDescendant = (componentId, visited = new Set()) => {
            if (visited.has(componentId)) return [];
            visited.add(componentId);
            
            const component = components[componentId];
            if (!component || !component.children_ids) return [];
            
            const visibleChildren = [];
            
            for (const childId of component.children_ids) {
                const child = components[childId];
                if (child && this.shouldShowComponent(child)) {
                    visibleChildren.push(childId);
                } else if (child) {
                    // Recursively find visible descendants
                    visibleChildren.push(...findNextVisibleDescendant(childId, visited));
                }
            }
            
            return visibleChildren;
        };
        
        Object.entries(components).forEach(([id, component]) => {
            // Skip if component is not shown
            if (!this.shouldShowComponent(component)) {
                return;
            }
            
            if (component.children_ids) {
                // Find all visible descendants (direct children or through hidden components)
                const visibleDescendants = findNextVisibleDescendant(id);
                
                visibleDescendants.forEach(descendantId => {
                    const startPos = positions[id];
                    const endPos = positions[descendantId];
                    
                    if (startPos && endPos) {
                        const connectionGroup = connectionsGroup.append('g')
                            .attr('class', 'connection-group');
                        
                        // Draw the connection path
                        const path = connectionGroup.append('path')
                            .attr('class', 'connection')
                            .attr('d', this.createPath(startPos, endPos))
                            .attr('fill', 'none')
                            .attr('stroke', '#444')
                            .attr('stroke-width', 2);
                        
                        // Add dimension label if in detailed view
                        const descendant = components[descendantId];
                        if (this.detailLevel === 'high' && component.output_dim && descendant.input_dim) {
                            const midX = (startPos.x + startPos.width/2 + endPos.x + endPos.width/2) / 2;
                            const midY = (startPos.y + startPos.height + endPos.y) / 2;
                            
                            // Create dimension label
                            let dimText = '';
                            if (component.output_dim && component.output_dim.length > 0) {
                                dimText = component.output_dim.join('×');
                            }
                            
                            if (dimText) {
                                connectionGroup.append('text')
                                    .attr('x', midX)
                                    .attr('y', midY)
                                    .attr('text-anchor', 'middle')
                                    .attr('class', 'dimension-label connection-dim')
                                    .attr('fill', '#666')
                                    .attr('font-size', '10px')
                                    .attr('dy', '-3')
                                    .text(dimText);
                            }
                        }
                        
                        // Store reference for animations
                        path.attr('data-from', id).attr('data-to', descendantId);
                    }
                });
            }
        });
    }
    
    createPath(start, end) {
        const startX = start.x + start.width / 2;
        const startY = start.y + start.height;
        const endX = end.x + end.width / 2;
        const endY = end.y;
        
        const midY = (startY + endY) / 2;
        
        return `M ${startX} ${startY} 
                C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`;
    }
    
    drawComponent(id, component, position) {
        if (!position) return;
        
        const group = this.mainGroup.append('g')
            .attr('class', `component component-${component.type}`)
            .attr('id', `component-${id}`)
            .attr('transform', `translate(${position.x}, ${position.y})`);
        
        let shape;
        
        // Special handling for transformer blocks - create a container
        if (component.type === 'transformer_block') {
            // Draw rounded rectangle container for transformer block
            shape = group.append('rect')
                .attr('width', position.width + 40)  // Extra width for padding
                .attr('height', position.height + 40)  // Extra height for padding
                .attr('x', -20)  // Offset for padding
                .attr('y', -20)  // Offset for padding
                .attr('rx', 15)
                .attr('ry', 15)
                .attr('fill', 'rgba(60, 60, 80, 0.15)')
                .attr('stroke', '#8b5cf6')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '5,5');
        }
        // Draw shapes based on component category
        else if (['matmul', 'add', 'residual_add', 'concat', 'split', 'softmax'].includes(component.type)) {
            // Operations are diamonds
            const cx = position.width / 2;
            const cy = position.height / 2;
            const size = Math.min(position.width, position.height) * 0.8;
            
            shape = group.append('path')
                .attr('d', `M ${cx} ${cy - size/2} L ${cx + size/2} ${cy} L ${cx} ${cy + size/2} L ${cx - size/2} ${cy} Z`);
                
            // Add operation symbol
            const symbols = {
                'add': '+',
                'residual_add': '⊕',
                'matmul': '×',
                'concat': '⊔',
                'split': '⊓',
                'softmax': 'σ'
            };
            
            if (symbols[component.type]) {
                group.append('text')
                    .attr('x', cx)
                    .attr('y', cy)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .attr('fill', 'white')
                    .attr('font-size', '16px')
                    .attr('font-weight', 'bold')
                    .text(symbols[component.type]);
            }
        }
        else if (['activation', 'dropout'].includes(component.type)) {
            // Activations are circles
            shape = group.append('circle')
                .attr('cx', position.width / 2)
                .attr('cy', position.height / 2)
                .attr('r', Math.min(position.width, position.height) / 2.5);
        }
        else {
            // Layers/weights are rectangles
            shape = group.append('rect')
                .attr('width', position.width)
                .attr('height', position.height)
                .attr('rx', 5)
                .attr('ry', 5);
        }
        
        if (shape && component.type !== 'transformer_block') {
            shape.attr('class', 'component-shape')
                .attr('fill', this.getComponentColor(component.type))
                .attr('stroke', '#333')
                .attr('stroke-width', 2);
        }
        
        // Add label
        if (!['add', 'residual_add', 'matmul', 'concat', 'split', 'softmax'].includes(component.type)) {
            const text = group.append('text')
                .attr('x', position.width / 2)
                .attr('y', position.height / 2)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .attr('class', 'component-label')
                .attr('fill', component.type === 'transformer_block' ? '#888' : 'white')
                .attr('font-size', component.type === 'transformer_block' ? '14px' : '12px')
                .attr('font-weight', component.type === 'transformer_block' ? 'bold' : 'normal');
            
            // Handle text for different component types
            if (component.type === 'attention_head') {
                text.text(`H${component.params.head_idx + 1}`);
            } else if (component.type === 'dropout') {
                text.text(`Drop ${component.params.p || 0.1}`);
            } else {
                // Split long names
                const words = component.name.split(' ');
                if (words.length > 2 && component.type !== 'transformer_block') {
                    text.append('tspan')
                        .attr('x', position.width / 2)
                        .attr('dy', '-0.3em')
                        .text(words.slice(0, -1).join(' '));
                    text.append('tspan')
                        .attr('x', position.width / 2)
                        .attr('dy', '1.2em')
                        .text(words[words.length - 1]);
                } else {
                    text.text(component.name);
                }
            }
        }
        
        // Add dimension labels for linear layers and embeddings
        if (['linear', 'embedding'].includes(component.type) && component.params) {
            const dimText = group.append('text')
                .attr('x', position.width / 2)
                .attr('y', position.height + 15)
                .attr('text-anchor', 'middle')
                .attr('class', 'dimension-label')
                .attr('fill', '#888')
                .attr('font-size', '10px');
                
            if (component.params.in_features && component.params.out_features) {
                dimText.text(`${component.params.in_features} → ${component.params.out_features}`);
            } else if (component.params.vocab_size && component.params.n_embed) {
                dimText.text(`${component.params.vocab_size} → ${component.params.n_embed}`);
            } else if (component.params.block_size && component.params.n_embed) {
                dimText.text(`${component.params.block_size} → ${component.params.n_embed}`);
            }
        }
        
        // Add parameter info for other components
        if (component.type === 'layer_norm' && component.params.n_embed) {
            group.append('text')
                .attr('x', position.width / 2)
                .attr('y', position.height + 15)
                .attr('text-anchor', 'middle')
                .attr('class', 'dimension-label')
                .attr('fill', '#888')
                .attr('font-size', '10px')
                .text(`[${component.params.n_embed}]`);
        }
        
        // Add hover effect
        group.on('mouseover', () => this.handleHover(id, true))
            .on('mouseout', () => this.handleHover(id, false))
            .on('click', () => this.handleClick(id));
        
        // Set initial state
        this.updateComponentState(id, component.state);
    }
    
    getComponentColor(type) {
        const colors = {
            // Layers and weights (blue/purple tones) - rectangles
            embedding: '#8b5cf6',  // Purple
            linear: '#3b82f6',     // Blue
            layer_norm: '#f59e0b', // Orange
            feed_forward: '#6366f1', // Indigo
            
            // Operations (green/cyan tones) - diamonds
            matmul: '#10b981',      // Emerald
            add: '#14b8a6',         // Teal
            residual_add: '#06b6d4', // Cyan
            concat: '#0891b2',      // Cyan dark
            split: '#0284c7',       // Sky
            softmax: '#059669',     // Green
            
            // Activations (red/pink tones) - circles
            activation: '#ec4899',  // Pink
            dropout: '#ef4444',     // Red
            
            // Complex components
            attention: '#10b981',   // Emerald
            attention_head: '#34d399', // Green light
            transformer_block: 'transparent', // No fill, just border
        };
        return colors[type] || '#6b7280';
    }
    
    updateComponentState(id, state) {
        const component = d3.select(`#component-${id}`);
        const shape = component.select('.component-shape');
        
        // Remove all state classes
        component.classed('state-inactive', false)
            .classed('state-forward', false)
            .classed('state-backward', false)
            .classed('state-computing', false)
            .classed('state-disabled', false);
        
        // Add new state class
        component.classed(`state-${state}`, true);
        
        // Apply visual effects based on state
        switch (state) {
            case 'forward':
                this.animateComponent(id, '#4ade80', 'forward');
                break;
            case 'backward':
                this.animateComponent(id, '#fb923c', 'backward');
                break;
            case 'computing':
                shape.attr('filter', 'url(#glow)');
                break;
            case 'disabled':
                shape.attr('opacity', 0.3);
                break;
            default:
                shape.attr('filter', null).attr('opacity', 1);
        }
    }
    
    animateComponent(id, color, direction) {
        const component = d3.select(`#component-${id}`);
        const shape = component.select('.component-shape');
        
        // Create unique animation ID
        const animId = `${id}-${Date.now()}`;
        this.activeAnimations.add(animId);
        
        // Pulse animation
        shape.transition()
            .duration(this.animationDuration)
            .attr('filter', 'url(#glow)')
            .style('fill', color)
            .transition()
            .duration(this.animationDuration)
            .attr('filter', null)
            .style('fill', this.getComponentColor(this.components[id].type))
            .on('end', () => {
                this.activeAnimations.delete(animId);
            });
        
        // Animate connections
        if (direction === 'forward') {
            this.animateConnections(id, 'from');
        } else if (direction === 'backward') {
            this.animateConnections(id, 'to');
        }
    }
    
    animateConnections(componentId, direction) {
        const connections = d3.selectAll(`.connection[data-${direction}="${componentId}"]`);
        
        connections.each(function() {
            const path = d3.select(this);
            const length = this.getTotalLength();
            
            // Create gradient for animation
            const gradient = d3.select('defs').append('linearGradient')
                .attr('id', `gradient-${componentId}-${Date.now()}`)
                .attr('gradientUnits', 'userSpaceOnUse');
            
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', '#444');
            
            gradient.append('stop')
                .attr('offset', '50%')
                .attr('stop-color', direction === 'from' ? '#4ade80' : '#fb923c');
            
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', '#444');
            
            // Animate gradient
            path.attr('stroke', `url(#${gradient.attr('id')})`);
            
            gradient.selectAll('stop')
                .transition()
                .duration(600)
                .attr('offset', function(d, i) {
                    return `${(i * 50 + 50) % 150}%`;
                })
                .on('end', function() {
                    path.attr('stroke', '#444');
                    gradient.remove();
                });
        });
    }
    
    handleHover(id, isHover) {
        const component = d3.select(`#component-${id}`);
        
        if (isHover) {
            component.classed('hover', true);
            // Highlight connected components
            this.highlightConnections(id);
        } else {
            component.classed('hover', false);
            this.clearHighlights();
        }
    }
    
    highlightConnections(id) {
        // Dim all components
        d3.selectAll('.component').classed('dimmed', true);
        
        // Highlight this component
        d3.select(`#component-${id}`).classed('dimmed', false).classed('highlighted', true);
        
        // Highlight parent and children
        const component = this.components[id];
        if (component) {
            if (component.parent_id) {
                d3.select(`#component-${component.parent_id}`).classed('dimmed', false);
            }
            component.children_ids.forEach(childId => {
                d3.select(`#component-${childId}`).classed('dimmed', false);
            });
        }
    }
    
    clearHighlights() {
        d3.selectAll('.component')
            .classed('dimmed', false)
            .classed('highlighted', false);
    }
    
    handleClick(id) {
        const component = this.components[id];
        if (component) {
            this.showComponentDetails(component);
        }
    }
    
    showComponentDetails(component) {
        const details = {
            name: component.name,
            type: component.type,
            params: component.params,
            state: component.state
        };
        
        // Emit custom event with component details
        const event = new CustomEvent('component-selected', { detail: details });
        this.container.dispatchEvent(event);
    }
    
    updateStates(stateUpdates) {
        Object.entries(stateUpdates).forEach(([id, state]) => {
            if (this.components[id]) {
                this.components[id].state = state;
                this.updateComponentState(id, state);
            }
        });
    }
    
    showVisualizationStatus(enabled) {
        if (enabled) {
            this.statusText.style('display', 'block');
            this.stepText.style('display', 'block');
        } else {
            this.statusText.style('display', 'none');
            this.stepText.style('display', 'none');
        }
    }
    
    updateTrainingStatus(phase, step, activeComponent) {
        if (phase) {
            this.statusText.text(`Phase: ${phase}`);
        }
        if (step !== undefined) {
            this.stepText.text(`Step: ${step}`);
        }
        
        // Highlight active component more prominently in visualization mode
        if (activeComponent && this.components[activeComponent]) {
            // Clear previous highlights
            d3.selectAll('.component').classed('viz-active', false);
            // Add strong highlight to active component
            d3.select(`#component-${activeComponent}`).classed('viz-active', true);
        }
    }
    
    setDetailLevel(level) {
        this.detailLevel = level;
        
        // Adjust layout spacing based on detail level
        if (level === 'low') {
            this.layout.verticalSpacing = 100;
            this.layout.horizontalSpacing = 180;
        } else {
            this.layout.verticalSpacing = 80;
            this.layout.horizontalSpacing = 150;
        }
        
        // Re-render the architecture with new detail level
        if (Object.keys(this.components).length > 0) {
            const architectureData = {
                components: this.components,
                root_components: Object.keys(this.components).filter(id => 
                    this.components[id].parent_id === null
                )
            };
            this.updateArchitecture(architectureData);
        }
    }
    
    shouldShowComponent(component) {
        if (this.detailLevel === 'high') {
            // Show all components in high detail mode
            return true;
        }
        
        // In low detail mode, hide certain internal components
        const hiddenTypes = [
            'dropout',
            'residual_add',
            'add',
            'split',
            'concat',
            'attention_head',
            'layer_norm'
        ];
        
        // Always show transformer blocks
        if (component.type === 'transformer_block') {
            return true;
        }
        
        // Always show major components
        const majorTypes = [
            'embedding',
            'linear',
            'attention',
            'feed_forward',
            'activation'
        ];
        
        if (majorTypes.includes(component.type)) {
            return true;
        }
        
        // Hide internal details in low detail mode
        return !hiddenTypes.includes(component.type);
    }
}

// Export for use in main.js
window.ArchitectureVisualizer = ArchitectureVisualizer;