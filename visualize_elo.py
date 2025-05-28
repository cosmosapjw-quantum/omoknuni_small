#!/usr/bin/env python3
"""
ELO Rating Visualization for AlphaZero Models

This script creates visualizations of ELO rating progression over training iterations.
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
from python.alphazero.elo_system import ELORatingSystem


def load_elo_data(elo_file: str) -> Tuple[Dict, Dict]:
    """Load ELO rating data from file."""
    with open(elo_file, 'r') as f:
        data = json.load(f)
    
    ratings = data['ratings']
    history = {}
    
    # Convert history timestamps
    if 'history' in data:
        for model_id, hist in data['history'].items():
            history[model_id] = [(datetime.fromisoformat(t), r) for t, r in hist]
    
    return ratings, history


def plot_elo_progression(history: Dict[str, List[Tuple[datetime, float]]], 
                        output_path: str = "elo_progression.png"):
    """Plot ELO rating progression over time."""
    plt.figure(figsize=(12, 8))
    
    # Filter out random policy unless specifically requested
    models_to_plot = {k: v for k, v in history.items() if k != "random_policy"}
    
    # Plot each model's progression
    for model_id, rating_history in models_to_plot.items():
        if len(rating_history) > 0:
            times = [t for t, _ in rating_history]
            ratings = [r for _, r in rating_history]
            
            # Use different styles for different model types
            if model_id == "initial":
                plt.plot(times, ratings, 'o-', label=model_id, linewidth=2, markersize=8)
            elif model_id.startswith("iter_"):
                # Extract iteration number for coloring
                try:
                    iter_num = int(model_id.split("_")[1])
                    color = plt.cm.viridis(iter_num / 20)  # Normalize to colormap
                    plt.plot(times, ratings, 'o-', label=model_id, 
                            linewidth=1.5, markersize=6, color=color, alpha=0.8)
                except:
                    plt.plot(times, ratings, 'o-', label=model_id, linewidth=1.5, markersize=6)
            else:
                plt.plot(times, ratings, 'o-', label=model_id, linewidth=1.5, markersize=6)
    
    # Add horizontal line at ELO 0 (random policy baseline)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random Policy (ELO 0)')
    
    # Add grid lines at major ELO milestones
    for elo in range(-500, 2500, 200):
        plt.axhline(y=elo, color='gray', linestyle=':', alpha=0.3)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('ELO Rating', fontsize=12)
    plt.title('AlphaZero Model ELO Rating Progression', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ELO progression plot to {output_path}")


def plot_elo_by_iteration(ratings: Dict[str, float], output_path: str = "elo_by_iteration.png"):
    """Plot ELO ratings by iteration number."""
    # Extract iteration models
    iter_models = []
    for model_id, rating in ratings.items():
        if model_id.startswith("iter_"):
            try:
                iter_num = int(model_id.split("_")[1])
                iter_models.append((iter_num, rating, model_id))
            except:
                pass
    
    # Sort by iteration
    iter_models.sort(key=lambda x: x[0])
    
    if not iter_models:
        print("No iteration models found to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    iterations = [x[0] for x in iter_models]
    elo_ratings = [x[1] for x in iter_models]
    
    # Plot main line
    plt.plot(iterations, elo_ratings, 'b-', linewidth=2, label='Model ELO')
    plt.scatter(iterations, elo_ratings, color='blue', s=100, zorder=5)
    
    # Add value labels
    for iter_num, rating, _ in iter_models:
        plt.annotate(f'{rating:.0f}', 
                    (iter_num, rating), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    # Add trend line
    if len(iterations) > 1:
        z = np.polyfit(iterations, elo_ratings, 1)
        p = np.poly1d(z)
        plt.plot(iterations, p(iterations), "r--", alpha=0.5, 
                label=f'Trend (slope: {z[0]:.1f} ELO/iter)')
    
    # Add horizontal reference lines
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random Policy')
    
    # Check if we have initial model rating
    if "initial" in ratings:
        plt.axhline(y=ratings["initial"], color='green', linestyle='--', 
                   alpha=0.5, label=f'Initial Model ({ratings["initial"]:.0f})')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('ELO Rating', fontsize=12)
    plt.title('ELO Rating by Training Iteration', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set integer x-axis
    if iterations:
        plt.xticks(range(min(iterations), max(iterations) + 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ELO by iteration plot to {output_path}")


def plot_elo_distribution(ratings: Dict[str, float], output_path: str = "elo_distribution.png"):
    """Plot distribution of current ELO ratings."""
    # Filter and sort ratings
    filtered_ratings = [(k, v) for k, v in ratings.items() if k != "random_policy"]
    filtered_ratings.sort(key=lambda x: x[1], reverse=True)
    
    if not filtered_ratings:
        print("No models to plot")
        return
    
    plt.figure(figsize=(10, max(6, len(filtered_ratings) * 0.4)))
    
    models = [x[0] for x in filtered_ratings]
    elos = [x[1] for x in filtered_ratings]
    
    # Create horizontal bar chart
    colors = []
    for model in models:
        if model == "initial":
            colors.append('green')
        elif model.startswith("iter_"):
            colors.append('blue')
        else:
            colors.append('gray')
    
    bars = plt.barh(models, elos, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (model, elo) in enumerate(filtered_ratings):
        plt.text(elo + 10, i, f'{elo:.0f}', va='center')
    
    # Add vertical line at ELO 0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Random Policy')
    
    plt.xlabel('ELO Rating', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Current ELO Ratings Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Initial Model'),
        Patch(facecolor='blue', alpha=0.7, label='Training Iterations'),
        Patch(facecolor='gray', alpha=0.7, label='Other Models')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ELO distribution plot to {output_path}")


def generate_elo_summary(ratings: Dict[str, float], history: Dict, 
                        output_path: str = "elo_summary.txt"):
    """Generate a text summary of ELO ratings."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AlphaZero ELO Rating Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Overall statistics
        model_ratings = [r for m, r in ratings.items() if m != "random_policy"]
        if model_ratings:
            f.write("Overall Statistics:\n")
            f.write(f"  Total models: {len(model_ratings)}\n")
            f.write(f"  Average ELO: {np.mean(model_ratings):.1f}\n")
            f.write(f"  Highest ELO: {max(model_ratings):.1f}\n")
            f.write(f"  Lowest ELO: {min(model_ratings):.1f}\n")
            f.write(f"  ELO Range: {max(model_ratings) - min(model_ratings):.1f}\n\n")
        
        # Top models
        sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        f.write("Top 10 Models:\n")
        for i, (model, rating) in enumerate(sorted_models[:10]):
            f.write(f"  {i+1:2d}. {model:<20} {rating:>7.1f}")
            if model != "random_policy":
                win_prob = 1.0 / (1.0 + 10**(-rating/400))
                f.write(f"  (vs random: {win_prob:.1%})")
            f.write("\n")
        
        # Progress analysis
        f.write("\nTraining Progress:\n")
        iter_models = [(int(m.split("_")[1]), r) for m, r in ratings.items() 
                      if m.startswith("iter_") and "_" in m]
        if iter_models:
            iter_models.sort()
            
            # Calculate improvement
            first_iter, first_elo = iter_models[0]
            last_iter, last_elo = iter_models[-1]
            
            f.write(f"  First iteration ({first_iter}): {first_elo:.1f}\n")
            f.write(f"  Last iteration ({last_iter}): {last_elo:.1f}\n")
            f.write(f"  Total improvement: {last_elo - first_elo:+.1f}\n")
            f.write(f"  Average per iteration: {(last_elo - first_elo)/(last_iter - first_iter):.1f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Saved ELO summary to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize AlphaZero ELO Ratings')
    parser.add_argument('--elo-file', type=str, default='models/elo_ratings.json',
                       help='Path to ELO ratings file')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save visualizations')
    parser.add_argument('--plots', type=str, nargs='+', 
                       default=['progression', 'iteration', 'distribution'],
                       choices=['progression', 'iteration', 'distribution', 'all'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Check if ELO file exists
    if not os.path.exists(args.elo_file):
        print(f"Error: ELO file not found: {args.elo_file}")
        sys.exit(1)
    
    # Load ELO data
    ratings, history = load_elo_data(args.elo_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate requested plots
    if 'all' in args.plots:
        args.plots = ['progression', 'iteration', 'distribution']
    
    if 'progression' in args.plots:
        output_path = os.path.join(args.output_dir, 'elo_progression.png')
        plot_elo_progression(history, output_path)
    
    if 'iteration' in args.plots:
        output_path = os.path.join(args.output_dir, 'elo_by_iteration.png')
        plot_elo_by_iteration(ratings, output_path)
    
    if 'distribution' in args.plots:
        output_path = os.path.join(args.output_dir, 'elo_distribution.png')
        plot_elo_distribution(ratings, output_path)
    
    # Always generate summary
    summary_path = os.path.join(args.output_dir, 'elo_summary.txt')
    generate_elo_summary(ratings, history, summary_path)
    
    print(f"\nVisualization complete! Files saved to {args.output_dir}")


if __name__ == '__main__':
    main()