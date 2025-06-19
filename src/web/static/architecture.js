class ArchitectureVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = null;
        this.components = {};
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
        
        // Draw components
        Object.entries(components).forEach(([id, component]) => {
            this.drawComponent(id, component, positions[id]);
        });
    }
    
    calculateLayout(components, rootIds) {
        const positions = {};
        let currentY = this.layout.padding;
        
        // Group components by layer
        const layers = this.groupByLayers(components, rootIds);
        
        layers.forEach((layer, layerIndex) => {
            const layerWidth = layer.length * this.layout.horizontalSpacing;
            const startX = (this.layout.width - layerWidth) / 2;
            
            layer.forEach((componentId, index) => {
                const component = components[componentId];
                let width = this.layout.nodeWidth;
                let height = this.layout.nodeHeight;
                
                // Adjust size based on component type
                if (component.type === 'attention_head') {
                    width = height = this.layout.headSize;
                }
                
                positions[componentId] = {
                    x: startX + index * this.layout.horizontalSpacing,
                    y: currentY,
                    width: width,
                    height: height
                };
            });
            
            currentY += this.layout.verticalSpacing;
        });
        
        // Update SVG height if needed
        const totalHeight = currentY + this.layout.padding;
        if (totalHeight > this.layout.height) {
            this.layout.height = totalHeight;
            this.svg.attr('viewBox', `0 0 ${this.layout.width} ${this.layout.height}`);
        }
        
        return positions;
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
        
        Object.entries(components).forEach(([id, component]) => {
            if (component.children_ids) {
                component.children_ids.forEach(childId => {
                    const startPos = positions[id];
                    const endPos = positions[childId];
                    
                    if (startPos && endPos) {
                        const path = connectionsGroup.append('path')
                            .attr('class', 'connection')
                            .attr('d', this.createPath(startPos, endPos))
                            .attr('fill', 'none')
                            .attr('stroke', '#444')
                            .attr('stroke-width', 2);
                        
                        // Store reference for animations
                        path.attr('data-from', id).attr('data-to', childId);
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
        
        // Draw shape based on component type
        switch (component.type) {
            case 'attention_head':
                shape = group.append('circle')
                    .attr('cx', position.width / 2)
                    .attr('cy', position.height / 2)
                    .attr('r', position.width / 2);
                break;
                
            case 'layer_norm':
            case 'dropout':
                shape = group.append('ellipse')
                    .attr('cx', position.width / 2)
                    .attr('cy', position.height / 2)
                    .attr('rx', position.width / 2)
                    .attr('ry', position.height / 2);
                break;
                
            case 'add':
                shape = group.append('circle')
                    .attr('cx', position.width / 2)
                    .attr('cy', position.height / 2)
                    .attr('r', 15);
                group.append('text')
                    .attr('x', position.width / 2)
                    .attr('y', position.height / 2)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .attr('fill', 'white')
                    .attr('font-size', '18px')
                    .text('+');
                break;
                
            default:
                shape = group.append('rect')
                    .attr('width', position.width)
                    .attr('height', position.height)
                    .attr('rx', 5)
                    .attr('ry', 5);
        }
        
        shape.attr('class', 'component-shape')
            .attr('fill', this.getComponentColor(component.type))
            .attr('stroke', '#333')
            .attr('stroke-width', 2);
        
        // Add label (except for add nodes)
        if (component.type !== 'add') {
            const text = group.append('text')
                .attr('x', position.width / 2)
                .attr('y', position.height / 2)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .attr('class', 'component-label')
                .attr('fill', 'white')
                .attr('font-size', '12px');
            
            // Split long names
            const words = component.name.split(' ');
            if (words.length > 2 && component.type !== 'attention_head') {
                text.append('tspan')
                    .attr('x', position.width / 2)
                    .attr('dy', '-0.3em')
                    .text(words.slice(0, -1).join(' '));
                text.append('tspan')
                    .attr('x', position.width / 2)
                    .attr('dy', '1.2em')
                    .text(words[words.length - 1]);
            } else {
                text.text(component.type === 'attention_head' ? `H${component.name.split(' ')[1]}` : component.name);
            }
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
            embedding: '#9333ea',
            linear: '#3b82f6',
            attention: '#10b981',
            attention_head: '#34d399',
            layer_norm: '#f59e0b',
            dropout: '#ef4444',
            activation: '#ec4899',
            add: '#6b7280',
            transformer_block: '#1f2937',
            feed_forward: '#6366f1'
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
}

// Export for use in main.js
window.ArchitectureVisualizer = ArchitectureVisualizer;