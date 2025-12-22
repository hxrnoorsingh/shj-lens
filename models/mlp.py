"""
Multi-Layer Perceptron for SHJ Category Learning
Small network with online/quasi-online training to match human learning
"""

import numpy as np
import json
from pathlib import Path

class MLPSHJ:
    def __init__(self, hidden_sizes=[8, 4], learning_rate=0.01, random_seed=None):
        """
        Initialize MLP
        
        Args:
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate for SGD
            random_seed: Random seed for reproducibility
        """
        self.hidden_sizes = hidden_sizes
        self.lr = learning_rate
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize network: 3 inputs -> hidden layers -> 1 output
        self.layers = []
        layer_sizes = [3] + hidden_sizes + [1]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.layers.append({
                'W': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale,
                'b': np.zeros(layer_sizes[i+1])
            })
        
        # Track learning history
        self.trial_history = []
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, x):
        """
        Forward pass
        
        Returns:
            output, activations (for backprop)
        """
        activations = [x]
        
        for i, layer in enumerate(self.layers):
            z = np.dot(activations[-1], layer['W']) + layer['b']
            
            # ReLU for hidden layers, sigmoid for output
            if i < len(self.layers) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            
            activations.append(a)
        
        return activations[-1], activations
    
    def backward(self, x, y_true, activations):
        """
        Backpropagation
        
        Args:
            x: Input features
            y_true: True label (0 or 1)
            activations: Activations from forward pass
        """
        # Convert category to binary
        y = 1 if y_true == 'A' else 0
        
        # Output layer gradient (sigmoid + BCE loss)
        output = activations[-1]
        delta = output - y
        
        # Backpropagate
        gradients = []
        for i in range(len(self.layers) - 1, -1, -1):
            # Compute gradients
            grad_W = np.outer(activations[i], delta)
            grad_b = delta
            
            gradients.insert(0, {'W': grad_W, 'b': grad_b})
            
            # Propagate to previous layer
            if i > 0:
                delta = np.dot(delta, self.layers[i]['W'].T)
                # Apply ReLU derivative
                z = np.dot(activations[i-1], self.layers[i-1]['W']) + self.layers[i-1]['b']
                delta = delta * self.relu_derivative(z)
        
        return gradients
    
    def update(self, x, y_true):
        """
        Online update (single trial)
        
        Args:
            x: Feature vector [shape, color, size]
            y_true: True category ('A' or 'B')
        
        Returns:
            accuracy (0 or 1)
        """
        # Forward pass
        output, activations = self.forward(x)
        
        # Backward pass
        gradients = self.backward(x, y_true, activations)
        
        # Update weights
        for i, layer in enumerate(self.layers):
            layer['W'] -= self.lr * gradients[i]['W']
            layer['b'] -= self.lr * gradients[i]['b']
        
        # Make prediction
        prediction = 'A' if output >= 0.5 else 'B'
        accuracy = 1 if prediction == y_true else 0
        
        # Record trial
        self.trial_history.append({
            'prediction': prediction,
            'true_category': y_true,
            'accuracy': accuracy,
            'probability_A': float(output)
        })
        
        return accuracy
    
    def predict_proba(self, x):
        """Predict probability of category A"""
        output, _ = self.forward(x)
        return float(output)
    
    def predict(self, x):
        """Predict category"""
        prob = self.predict_proba(x)
        return 'A' if prob >= 0.5 else 'B'
    
    def get_trial_history(self):
        """Return trial-by-trial history"""
        return self.trial_history
    
    def reset_history(self):
        """Reset trial history for new block"""
        self.trial_history = []
    
    def save_weights(self, filepath):
        """Save model weights"""
        np.savez(filepath, layers=[layer for layer in self.layers])
    
    def load_weights(self, filepath):
        """Load model weights"""
        data = np.load(filepath, allow_pickle=True)
        self.layers = data['layers'].tolist()


def train_on_shj_type(shj_type, stimuli, shj_mappings, model, max_trials=160, 
                       window_size=20, criterion=16, verbose=True):
    """
    Train MLP on specific SHJ type with same stopping criterion as humans
    
    Args:
        shj_type: Type key (e.g., 'type_1')
        stimuli: List of stimuli
        shj_mappings: Category mappings
        model: MLPSHJ instance
        max_trials: Maximum trials
        window_size: Rolling window size
        criterion: Correct trials needed in window
        verbose: Print progress
    
    Returns:
        Trial data dictionary
    """
    model.reset_history()
    
    categories = shj_mappings[shj_type]['categories']
    trial_data = []
    recent_responses = []
    
    trial = 0
    while trial < max_trials:
        trial += 1
        
        # Randomly sample stimulus (with replacement)
        stimulus = stimuli[np.random.randint(len(stimuli))]
        
        # Extract features
        x = np.array([stimulus['shape'], stimulus['color'], stimulus['size']], dtype=float)
        
        # Get true category
        true_category = categories[str(stimulus['id'])]
        
        # Predict (before update)
        prediction = model.predict(x)
        accuracy = 1 if prediction == true_category else 0
        
        # Update model (online learning)
        model.update(x, true_category)
        
        # Track for stopping criterion
        recent_responses.append(accuracy)
        if len(recent_responses) > window_size:
            recent_responses.pop(0)
        
        # Log trial
        trial_data.append({
            'trial': trial,
            'stimulus_id': stimulus['id'],
            'features': x.tolist(),
            'true_category': true_category,
            'prediction': prediction,
            'accuracy': accuracy
        })
        
        # Check stopping criterion
        if len(recent_responses) >= window_size:
            correct_count = sum(recent_responses)
            if correct_count >= criterion:
                if verbose:
                    print(f"Criterion reached at trial {trial}: {correct_count}/{window_size}")
                break
    
    if verbose:
        total_accuracy = sum(t['accuracy'] for t in trial_data) / len(trial_data)
        print(f"Completed {shj_type}: {trial} trials, {total_accuracy*100:.1f}% accuracy")
    
    return {
        'shj_type': shj_type,
        'trials': trial_data,
        'trials_to_criterion': trial if trial < max_trials else None
    }


if __name__ == '__main__':
    # Example usage
    print("Loading SHJ data...")
    
    with open('../experiment/stimuli.json', 'r') as f:
        stimuli = json.load(f)
    
    with open('../experiment/shj_mappings.json', 'r') as f:
        shj_mappings = json.load(f)
    
    print("\nTraining MLP on SHJ Types...\n")
    
    results = {}
    for shj_type in ['type_1', 'type_2', 'type_6']:
        print(f"\n{'='*50}")
        print(f"Training on {shj_mappings[shj_type]['name']}")
        print(f"{'='*50}")
        
        # Create fresh model for each type
        model = MLPSHJ(hidden_sizes=[8, 4], learning_rate=0.05, random_seed=42)
        
        # Train
        result = train_on_shj_type(
            shj_type, stimuli, shj_mappings, model,
            max_trials=160, window_size=20, criterion=16
        )
        
        results[shj_type] = result
    
    # Save results
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'mlp_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Results saved to ../results/mlp_results.json")
    print("="*50)
