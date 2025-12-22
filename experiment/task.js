// SHJ Category Learning Task
// Implements 16/20 rolling window stopping criterion

class SHJExperiment {
    constructor() {
        this.participantId = null;
        this.stimuli = [];
        this.shjMappings = {};
        this.currentBlock = 0;
        this.blockOrder = ['type_1', 'type_2', 'type_6']; // Counterbalanced in practice
        this.completedBlocks = [];

        // Trial data
        this.allTrialData = [];
        this.currentTrialData = [];
        this.currentTrial = 0;
        this.currentStimulus = null;
        this.trialStartTime = 0;

        // Practice
        this.isPractice = false;
        this.practiceTrials = 0;
        this.maxPracticeTrials = 8;

        // Performance tracking
        this.recentResponses = []; // For rolling window
        this.windowSize = 20;
        this.criterionThreshold = 16; // 16 out of 20 correct
        this.maxTrials = 160;

        this.init();
    }

    async init() {
        await this.loadData();
        this.setupEventListeners();
    }

    async loadData() {
        try {
            const [stimuliResponse, mappingsResponse] = await Promise.all([
                fetch('stimuli.json'),
                fetch('shj_mappings.json')
            ]);

            this.stimuli = await stimuliResponse.json();
            this.shjMappings = await mappingsResponse.json();

            console.log('Loaded stimuli:', this.stimuli);
            console.log('Loaded mappings:', this.shjMappings);
        } catch (error) {
            console.error('Error loading data:', error);
            alert('Error loading experiment data. Please refresh the page.');
        }
    }

    setupEventListeners() {
        // Start button
        document.getElementById('start-button').addEventListener('click', () => {
            const idInput = document.getElementById('participant-id');
            if (idInput.value.trim() === '') {
                alert('Please enter your participant ID');
                return;
            }
            this.participantId = idInput.value.trim();
            this.startPractice();
        });

        // Practice response buttons
        document.querySelectorAll('#practice-screen .btn-category').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleResponse(e.target.dataset.category, true));
        });

        // Main task response buttons
        document.querySelectorAll('#task-screen .btn-category').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleResponse(e.target.dataset.category, false));
        });

        // Download button
        document.getElementById('download-button').addEventListener('click', () => this.downloadData());

        // Next block button
        document.getElementById('next-block-button').addEventListener('click', () => this.startNextBlock());
    }

    startPractice() {
        this.isPractice = true;
        this.practiceTrials = 0;
        this.showScreen('practice-screen');
        this.presentStimulus(true);
    }

    startBlock(blockIndex) {
        this.currentBlock = blockIndex;
        this.currentTrial = 0;
        this.currentTrialData = [];
        this.recentResponses = [];

        const blockType = this.blockOrder[blockIndex];
        const blockName = this.shjMappings[blockType].name;

        document.getElementById('task-title').textContent = `Learning Block ${blockIndex + 1}: ${blockName}`;

        this.showScreen('task-screen');
        this.presentStimulus(false);
    }

    presentStimulus(isPractice) {
        // Randomly select a stimulus
        const stimulus = this.stimuli[Math.floor(Math.random() * this.stimuli.length)];
        this.currentStimulus = stimulus;
        this.trialStartTime = performance.now();

        // Clear feedback
        const feedbackEl = isPractice
            ? document.getElementById('practice-feedback')
            : document.getElementById('feedback');
        feedbackEl.textContent = '';
        feedbackEl.className = 'feedback';

        // Render stimulus
        this.renderStimulus(stimulus, isPractice);

        // Update trial counter
        if (isPractice) {
            this.practiceTrials++;
            document.getElementById('practice-trial-count').textContent =
                `Trial ${this.practiceTrials} of ${this.maxPracticeTrials}`;
        } else {
            this.currentTrial++;
            document.getElementById('trial-count').textContent = `Trial ${this.currentTrial}`;

            // Update progress bar
            const progress = Math.min(100, (this.currentTrial / this.maxTrials) * 100);
            document.getElementById('progress-fill').style.width = `${progress}%`;

            // Update accuracy display
            if (this.currentTrialData.length > 0) {
                const accuracy = this.currentTrialData.filter(t => t.accuracy === 1).length / this.currentTrialData.length;
                document.getElementById('accuracy-display').textContent =
                    `Accuracy: ${(accuracy * 100).toFixed(1)}%`;
            }
        }

        // Enable response buttons
        this.setButtonsEnabled(true, isPractice);
    }

    renderStimulus(stimulus, isPractice) {
        const svgId = isPractice ? 'practice-stimulus' : 'stimulus';
        const svg = document.getElementById(svgId);
        svg.innerHTML = ''; // Clear previous

        const size = stimulus.size_name === 'small' ? 80 : 140;
        const color = stimulus.color_name === 'red' ? '#e53e3e' : '#3182ce';
        const centerX = 150;
        const centerY = 150;

        if (stimulus.shape_name === 'circle') {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', centerX);
            circle.setAttribute('cy', centerY);
            circle.setAttribute('r', size / 2);
            circle.setAttribute('fill', color);
            circle.setAttribute('stroke', '#2d3748');
            circle.setAttribute('stroke-width', '3');
            svg.appendChild(circle);
        } else {
            const triangle = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            const height = size * 0.866; // equilateral triangle
            const points = [
                [centerX, centerY - height / 2],
                [centerX - size / 2, centerY + height / 2],
                [centerX + size / 2, centerY + height / 2]
            ].map(p => p.join(',')).join(' ');
            triangle.setAttribute('points', points);
            triangle.setAttribute('fill', color);
            triangle.setAttribute('stroke', '#2d3748');
            triangle.setAttribute('stroke-width', '3');
            svg.appendChild(triangle);
        }
    }

    handleResponse(response, isPractice) {
        const reactionTime = performance.now() - this.trialStartTime;

        // Disable buttons
        this.setButtonsEnabled(false, isPractice);

        // Get correct category
        const blockType = this.blockOrder[this.currentBlock];
        const correctCategory = this.shjMappings[blockType].categories[this.currentStimulus.id];
        const isCorrect = response === correctCategory;

        // Show feedback
        this.showFeedback(isCorrect, isPractice);

        if (!isPractice) {
            // Log trial data
            const trialData = {
                participant_id: this.participantId,
                shj_type: blockType,
                trial_number: this.currentTrial,
                stimulus_id: this.currentStimulus.id,
                stimulus_features: {
                    shape: this.currentStimulus.shape,
                    color: this.currentStimulus.color,
                    size: this.currentStimulus.size
                },
                stimulus_label: this.currentStimulus.label,
                correct_category: correctCategory,
                participant_response: response,
                accuracy: isCorrect ? 1 : 0,
                reaction_time_ms: Math.round(reactionTime)
            };

            this.currentTrialData.push(trialData);
            this.recentResponses.push(isCorrect ? 1 : 0);

            // Keep only last windowSize responses
            if (this.recentResponses.length > this.windowSize) {
                this.recentResponses.shift();
            }

            // Check stopping criterion
            setTimeout(() => {
                if (this.checkStoppingCriterion()) {
                    this.endBlock();
                } else {
                    this.presentStimulus(false);
                }
            }, 1500);
        } else {
            // Practice continues for fixed trials
            setTimeout(() => {
                if (this.practiceTrials >= this.maxPracticeTrials) {
                    this.endPractice();
                } else {
                    this.presentStimulus(true);
                }
            }, 1200);
        }
    }

    showFeedback(isCorrect, isPractice) {
        const feedbackEl = isPractice
            ? document.getElementById('practice-feedback')
            : document.getElementById('feedback');

        feedbackEl.textContent = isCorrect ? '✓ Correct!' : '✗ Incorrect';
        feedbackEl.className = isCorrect ? 'feedback correct' : 'feedback incorrect';
    }

    setButtonsEnabled(enabled, isPractice) {
        const selector = isPractice ? '#practice-screen .btn-category' : '#task-screen .btn-category';
        document.querySelectorAll(selector).forEach(btn => {
            btn.disabled = !enabled;
        });
    }

    checkStoppingCriterion() {
        // 16/20 rolling window OR max trials reached
        if (this.currentTrial >= this.maxTrials) {
            return true;
        }

        if (this.recentResponses.length >= this.windowSize) {
            const correctCount = this.recentResponses.reduce((a, b) => a + b, 0);
            if (correctCount >= this.criterionThreshold) {
                return true;
            }
        }

        return false;
    }

    endPractice() {
        alert('Practice complete! Now we\'ll begin the real experiment.');
        this.startBlock(0);
    }

    endBlock() {
        // Save block data
        this.allTrialData.push(...this.currentTrialData);
        this.completedBlocks.push(this.blockOrder[this.currentBlock]);

        // Calculate stats
        const accuracy = this.currentTrialData.filter(t => t.accuracy === 1).length / this.currentTrialData.length;
        const trialsCompleted = this.currentTrialData.length;

        const summaryText = `Block ${this.currentBlock + 1} Complete!\n\nTrials: ${trialsCompleted}\nAccuracy: ${(accuracy * 100).toFixed(1)}%`;

        document.getElementById('summary-stats').textContent = summaryText;

        // Check if more blocks remain
        if (this.currentBlock < this.blockOrder.length - 1) {
            const nextBlockType = this.blockOrder[this.currentBlock + 1];
            const nextBlockName = this.shjMappings[nextBlockType].name;
            document.getElementById('next-block-info').textContent =
                `Next block: ${nextBlockName}`;
            document.getElementById('next-section').style.display = 'block';
        } else {
            document.getElementById('next-section').style.display = 'none';
        }

        this.showScreen('completion-screen');
    }

    startNextBlock() {
        this.currentBlock++;
        this.startBlock(this.currentBlock);
    }

    downloadData() {
        const dataStr = JSON.stringify({
            participant_id: this.participantId,
            completed_blocks: this.completedBlocks,
            block_order: this.blockOrder,
            timestamp: new Date().toISOString(),
            stopping_criterion: '16/20 rolling window',
            max_trials: this.maxTrials,
            trials: this.allTrialData
        }, null, 2);

        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `shj_data_${this.participantId}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(screenId).classList.add('active');
    }
}

// Initialize experiment when page loads
window.addEventListener('DOMContentLoaded', () => {
    new SHJExperiment();
});
