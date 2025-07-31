#!/usr/bin/env python
"""
Example usage of the research experiments framework
Demonstrates how to run experiments and analyze results
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_quick_experiment():
    """Run a quick experiment with reduced epochs for demonstration"""
    print("Running quick experiment with 3 models...")
    
    # Check if dataset exists
    dataset_path = Path("./dataset")
    if not dataset_path.exists():
        print("❌ Dataset not found at ./dataset")
        print("Please download and extract the MIMII dataset first.")
        print("See README_RESEARCH.md for dataset structure requirements.")
        return False
    
    # Run experiment with reduced epochs for speed
    cmd = [
        sys.executable, "research_experiments.py",
        "--base_dir", "./dataset",
        "--model", "simple_ae",  # Just test one model for demo
        "--epochs", "5",  # Reduced for quick demo
        "--output_csv", "demo_results.csv"
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            return True
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Experiment timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False

def analyze_results():
    """Analyze and visualize experiment results"""
    results_file = Path("research_results/demo_results.csv")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return False
    
    print(f"📊 Analyzing results from {results_file}")
    
    try:
        # Load results
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} experiment results")
        
        # Display basic statistics
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        
        # Group by model and show statistics
        if 'model' in df.columns:
            model_stats = df.groupby('model').agg({
                'auc': ['mean', 'std', 'count'],
                'training_time_sec': 'mean',
                'model_size_mb': 'mean',
                'num_parameters': 'mean'
            }).round(4)
            
            print("\nModel Performance Summary:")
            print(model_stats)
        
        # Show individual results
        print(f"\nIndividual Results:")
        key_columns = ['model', 'machine_type', 'machine_id', 'auc', 'pr_auc', 'training_time_sec']
        available_columns = [col for col in key_columns if col in df.columns]
        print(df[available_columns].to_string(index=False))
        
        # Create visualizations if matplotlib is available
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('MIMII Experiment Results Analysis', fontsize=16)
            
            # AUC distribution
            if 'auc' in df.columns:
                axes[0, 0].hist(df['auc'], bins=10, alpha=0.7, color='skyblue')
                axes[0, 0].set_title('AUC Score Distribution')
                axes[0, 0].set_xlabel('AUC')
                axes[0, 0].set_ylabel('Frequency')
            
            # Training time vs AUC
            if 'training_time_sec' in df.columns and 'auc' in df.columns:
                axes[0, 1].scatter(df['training_time_sec'], df['auc'], alpha=0.7, color='orange')
                axes[0, 1].set_title('Training Time vs AUC')
                axes[0, 1].set_xlabel('Training Time (seconds)')
                axes[0, 1].set_ylabel('AUC')
            
            # Model size vs parameters
            if 'model_size_mb' in df.columns and 'num_parameters' in df.columns:
                axes[1, 0].scatter(df['num_parameters'], df['model_size_mb'], alpha=0.7, color='green')
                axes[1, 0].set_title('Model Parameters vs Size')
                axes[1, 0].set_xlabel('Number of Parameters')
                axes[1, 0].set_ylabel('Model Size (MB)')
            
            # Performance by machine type
            if 'machine_type' in df.columns and 'auc' in df.columns:
                df.boxplot(column='auc', by='machine_type', ax=axes[1, 1])
                axes[1, 1].set_title('AUC by Machine Type')
                axes[1, 1].set_xlabel('Machine Type')
                axes[1, 1].set_ylabel('AUC')
            
            plt.tight_layout()
            plt.savefig('research_results/experiment_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n📈 Visualizations saved to: research_results/experiment_analysis.png")
            
        except ImportError:
            print("\n⚠️  Matplotlib not available, skipping visualizations")
        except Exception as e:
            print(f"\n⚠️  Error creating visualizations: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False

def demonstrate_model_comparison():
    """Show how to compare multiple models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON EXAMPLE")
    print("="*60)
    
    print("""
To compare multiple models, you can run:

# Compare 3 different autoencoders
python research_experiments.py --base_dir ./dataset --model simple_ae --output_csv simple_ae_results.csv
python research_experiments.py --base_dir ./dataset --model deep_ae --output_csv deep_ae_results.csv  
python research_experiments.py --base_dir ./dataset --model vae --output_csv vae_results.csv

# Or run all models at once
python research_experiments.py --base_dir ./dataset --model all

# Then analyze with pandas:
import pandas as pd

# Load results
df = pd.read_csv('research_results/research_results.csv')

# Compare models
comparison = df.groupby('model').agg({
    'auc': ['mean', 'std'],
    'training_time_sec': 'mean',
    'model_size_mb': 'mean'
}).round(4)

print(comparison)
""")

def main():
    print("="*60)
    print("MIMII RESEARCH EXPERIMENTS - EXAMPLE USAGE")
    print("="*60)
    
    print("This script demonstrates how to:")
    print("1. Run a quick experiment")
    print("2. Analyze the results")
    print("3. Create visualizations")
    print("4. Compare multiple models")
    
    # Check if we should run the actual experiment
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("\n🚀 Running actual experiment...")
        
        if run_quick_experiment():
            print("\n📊 Analyzing results...")
            analyze_results()
        else:
            print("\n❌ Experiment failed, skipping analysis")
    else:
        print("\n💡 To run an actual experiment, use: python example_usage.py --run")
        print("   (Make sure you have the MIMII dataset in ./dataset/ first)")
    
    # Always show the comparison example
    demonstrate_model_comparison()
    
    print("\n" + "="*60)
    print("For more details, see README_RESEARCH.md")
    print("="*60)

if __name__ == "__main__":
    main()
