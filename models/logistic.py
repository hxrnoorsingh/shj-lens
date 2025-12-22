"""
Logistic Regression Model for SHJ Category Learning
Online learning with trial-by-trial updates to match human learning dynamics
"""

import numpy as np
import json
from pathlib import Path

class LogisticRegressionSHJ:
    def __init__(self, learning_rate=0.01, random_seed=None):
        """
        Initialize logistic regression model
        
        Args:
            learning_rate: Learning rate for SGD
            random_seed: Random seed for reproducibility
        """
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights for 3 features
        self.weights = np.random.randn(3) * 0.01
        self.bias = 0.0
        
        # Track learning history
        self.trial_history = []
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def predict_proba(self, x):
        """
        Predict probability of category A
        
        Args:
            x: Feature vector (shape, color, size)
        
        Returns:
            Probability of category A
        """
        z = np.dot(self.weights, x) + self.bias
        return self.sigmoid(z)
    
    def predict(self, x, threshold=0.5):
        """
        Predict category (binary)
        
        Args:
            x: Feature vector
            threshold: Decision threshold
        
        Returns:
            'A' if prob >= threshold, else 'B'
        """
        prob = self.predict_proba(x)
        return 'A' if prob >= threshold else 'B'
    
    def update(self, x, y_true):
        """
        Online SGD update (single trial)
        
        Args:
            x: Feature vector
            y_true: True category ('A' or 'B')
        """
        # Convert category to binary (A=1, B=0)
        y = 1 if y_true == 'A' else 0
        
        # Forward pass
        prob = self.predict_proba(x)
        
        # Compute gradient
        error = prob - y
        
        # Update weights and bias
        self.weights -= self.lr * error * x
        self.bias -= self.lr * error
        
        # Record trial
        prediction = self.predict(x)
        accuracy = 1 if prediction == y_true else 0
        
        self.trial_history.append({
            'prediction': prediction,
            'true_category': y_true,
            'accuracy': accuracy,
            'probability_A': float(prob)
        })
        
        return accuracy
    
    def get_trial_history(self):
        """Return trial-by-trial history"""
        return self.trial_history
    
    def reset_history(self):
        """Reset trial history for new block"""
        self.trial_history = []
    
    def save_weights(self, filepath):
        """Save model weights"""
        np.savez(filepath, weights=self.weights, bias=self.bias)
    
    def load_weights(self, filepath):
        """Load model weights"""
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']


def train_on_shj_type(shj_type, stimuli, shj_mappings, model, max_trials=160, 
                       window_size=20, criterion=16, verbose=True):
    """
    Train model on specific SHJ type with same stopping criterion as humans
    
    Args:
        shj_type: Type key (e.g., 'type_1')
        stimuli: List of stimuli
        shj_mappings: Category mappings
        model: LogisticRegressionSHJ instance
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
        
        # Randomly sample stimulus (with replacement, like humans)
        stimulus = stimuli[np.random.randint(len(stimuli))]
        
        # Extract features
        x = np.array([stimulus['shape'], stimulus['color'], stimulus['size']])
        
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
    
    print("\nTraining Logistic Regression on SHJ Types...\n")
    
    results = {}
    for shj_type in ['type_1', 'type_2', 'type_6']:
        print(f"\n{'='*50}")
        print(f"Training on {shj_mappings[shj_type]['name']}")
        print(f"{'='*50}")
        
        # Create fresh model for each type
        model = LogisticRegressionSHJ(learning_rate=0.1, random_seed=42)
        
        # Train
        result = train_on_shj_type(
            shj_type, stimuli, shj_mappings, model,
            max_trials=160, window_size=20, criterion=16
        )
        
        results[shj_type] = result
    
    # Save results
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'logistic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Results saved to ../results/logistic_results.json")
    print("="*50)
