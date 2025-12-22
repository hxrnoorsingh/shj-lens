"""
Unified training script for all models on all SHJ types
Runs multiple replications and saves aggregated results
"""

import json
import numpy as np
from pathlib import Path
from logistic import LogisticRegressionSHJ, train_on_shj_type as train_logistic
from mlp import MLPSHJ, train_on_shj_type as train_mlp

def run_multiple_replications(model_class, model_params, shj_type, stimuli, 
                               shj_mappings, n_reps=10, **training_params):
    """
    Run multiple replications of model training
    
    Returns:
        List of results from each replication
    """
    results = []
    
    for rep in range(n_reps):
        # Create model with different seed
        params = model_params.copy()
        params['random_seed'] = rep
        model = model_class(**params)
        
        # Train
        result = (train_logistic if model_class == LogisticRegressionSHJ else train_mlp)(
            shj_type, stimuli, shj_mappings, model,
            verbose=False, **training_params
        )
        
        results.append(result)
        
        # Print progress
        trials = result['trials_to_criterion']
        if trials:
            print(f"  Rep {rep+1}/{n_reps}: {trials} trials")
        else:
            print(f"  Rep {rep+1}/{n_reps}: Did not reach criterion")
    
    return results

def aggregate_results(results):
    """Aggregate results across replications"""
    trials_to_criterion = [r['trials_to_criterion'] for r in results if r['trials_to_criterion']]
    
    # Get maximum trial count for alignment
    max_trials = max(len(r['trials']) for r in results)
    
    # Create trial-by-trial accuracy matrix
    accuracy_matrix = []
    for result in results:
        trial_accs = [t['accuracy'] for t in result['trials']]
        # Pad with final accuracy if needed
        if len(trial_accs) < max_trials:
            trial_accs += [trial_accs[-1]] * (max_trials - len(trial_accs))
        accuracy_matrix.append(trial_accs)
    
    accuracy_matrix = np.array(accuracy_matrix)
    
    return {
        'trials_to_criterion': {
            'mean': float(np.mean(trials_to_criterion)) if trials_to_criterion else None,
            'median': float(np.median(trials_to_criterion)) if trials_to_criterion else None,
            'std': float(np.std(trials_to_criterion)) if trials_to_criterion else None,
            'n_reached': len(trials_to_criterion),
            'n_total': len(results)
        },
        'learning_curve': {
            'mean': accuracy_matrix.mean(axis=0).tolist(),
            'std': accuracy_matrix.std(axis=0).tolist(),
            'trials': list(range(1, max_trials + 1))
        }
    }

def main():
    print("="*70)
    print("SHJ Category Learning: Model Training")
    print("="*70)
    
    # Load data
    print("\nLoading experiment data...")
    with open('../experiment/stimuli.json', 'r') as f:
        stimuli = json.load(f)
    
    with open('../experiment/shj_mappings.json', 'r') as f:
        shj_mappings = json.load(f)
    
    # Training parameters
    training_params = {
        'max_trials': 160,
        'window_size': 20,
        'criterion': 16
    }
    
    n_reps = 10
    shj_types = ['type_1', 'type_2', 'type_6']
    
    all_results = {}
    
    # Train Logistic Regression
    print(f"\n{'='*70}")
    print("LOGISTIC REGRESSION")
    print(f"{'='*70}")
    
    all_results['logistic'] = {}
    for shj_type in shj_types:
        print(f"\n{shj_mappings[shj_type]['name']}")
        print("-" * 50)
        
        results = run_multiple_replications(
            LogisticRegressionSHJ,
            {'learning_rate': 0.1},
            shj_type, stimuli, shj_mappings,
            n_reps=n_reps,
            **training_params
        )
        
        all_results['logistic'][shj_type] = aggregate_results(results)
    
    # Train MLP
    print(f"\n{'='*70}")
    print("MULTI-LAYER PERCEPTRON")
    print(f"{'='*70}")
    
    all_results['mlp'] = {}
    for shj_type in shj_types:
        print(f"\n{shj_mappings[shj_type]['name']}")
        print("-" * 50)
        
        results = run_multiple_replications(
            MLPSHJ,
            {'hidden_sizes': [8, 4], 'learning_rate': 0.05},
            shj_type, stimuli, shj_mappings,
            n_reps=n_reps,
            **training_params
        )
        
        all_results['mlp'][shj_type] = aggregate_results(results)
    
    # Save results
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: Median Trials to Criterion")
    print(f"{'='*70}\n")
    
    print(f"{'SHJ Type':<30} {'Logistic':<15} {'MLP':<15}")
    print("-" * 60)
    
    for shj_type in shj_types:
        type_name = shj_mappings[shj_type]['name']
        log_median = all_results['logistic'][shj_type]['trials_to_criterion']['median']
        mlp_median = all_results['mlp'][shj_type]['trials_to_criterion']['median']
        
        log_str = f"{log_median:.0f}" if log_median else "N/A"
        mlp_str = f"{mlp_median:.0f}" if mlp_median else "N/A"
        
        print(f"{type_name:<30} {log_str:<15} {mlp_str:<15}")
    
    print(f"\n{'='*70}")
    print("Results saved to ../results/model_results.json")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
