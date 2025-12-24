#!/bin/bash

# Single Domain Training for Cross-Domain Few-Shot Text Classification


cuda=0
comment="SingleDomain"


BERT_PATH="./models/bert-base-uncased"
DATA_BASE_PATH="data"
RESULT_BASE_PATH="cross_domain_results"


mkdir -p $RESULT_BASE_PATH
mkdir -p $RESULT_BASE_PATH/plots

echo "=========================================="
echo "Single Domain Training with Visualization"
echo "=========================================="


cat > $RESULT_BASE_PATH/plot_training_curves.py << 'EOF'
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


try:
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8')
except Exception:
    try:
        plt.style.use('seaborn')
    except Exception:
        plt.style.use('default')
        print("Using default matplotlib style")

def plot_training_curves(log_file, output_dir, experiment_name):
    """training vurve"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    domain_losses = []
    adversarial_losses = []
    episodes = []

    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if 'episode' in data:
                            episodes.append(data['episode'])
                            
                            train_losses.append(float(data.get('train_loss', 0)))
                            train_accs.append(float(data.get('train_accuracy', 0)))
                            val_losses.append(float(data.get('val_loss', 0)))
                            val_accs.append(float(data.get('val_accuracy', 0)))
                            domain_losses.append(float(data.get('domain_loss', 0)))
                            adversarial_losses.append(float(data.get('adversarial_loss', 0)))
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        print(f"Error parsing line: {line.strip()[:50]}... Error: {e}")
                        continue
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if not episodes:
        print("No valid training data found")
        return

    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16, fontweight='bold')

    
    axes[0, 0].plot(episodes, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    axes[0, 0].plot(episodes, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    
    axes[0, 1].plot(episodes, train_accs, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
    axes[0, 1].plot(episodes, val_accs, 'r-', label='Val Accuracy', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    
    if any(d > 0 for d in domain_losses):
        axes[0, 2].plot(episodes, domain_losses, 'g-', label='Domain Loss', linewidth=2, alpha=0.8)
        axes[0, 2].set_title('Domain Loss', fontweight='bold')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Domain Loss\n(Single Domain Training)',
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Domain Loss (Not Used)', fontweight='bold')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)

    
    if any(a > 0 for a in adversarial_losses):
        axes[1, 0].plot(episodes, adversarial_losses, 'm-', label='Adversarial Loss', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Adversarial Loss', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Adversarial Loss\n(Single Domain Training)',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Adversarial Loss (Not Used)', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    
    if len(episodes) > 10:
        window = max(1, len(episodes) // 20)  
        smooth_train_loss = pd.Series(train_losses).rolling(window, center=True).mean()
        smooth_val_loss = pd.Series(val_losses).rolling(window, center=True).mean()

        axes[1, 1].plot(episodes, smooth_train_loss, 'b-', label='Smooth Train Loss', linewidth=2)
        axes[1, 1].plot(episodes, smooth_val_loss, 'r-', label='Smooth Val Loss', linewidth=2)
        axes[1, 1].set_title('Smoothed Loss Curves', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    
    axes[1, 2].axis('off')
    if train_accs and val_accs and train_losses and val_losses:
        stats_text = f"""Training Statistics:

Max Train Acc: {max(train_accs):.4f}
Max Val Acc: {max(val_accs):.4f}
Final Train Acc: {train_accs[-1]:.4f}
Final Val Acc: {val_accs[-1]:.4f}

Min Train Loss: {min(train_losses):.4f}
Min Val Loss: {min(val_losses):.4f}
Final Train Loss: {train_losses[-1]:.4f}
Final Val Loss: {val_losses[-1]:.4f}

Total Episodes: {len(episodes)}
        """
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    
    plot_path = os.path.join(output_dir, f'{experiment_name}_training_curves.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

def create_comparison_plot(result_dir):
    """create figure"""
    all_results = []

    
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith('_results.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        all_results.append({
                            'experiment': os.path.basename(root),
                            'accuracy': data.get('test_accuracy', 0),
                            'loss': data.get('test_loss', 0)
                        })
                except:
                    continue

    if not all_results:
        print("No results found for comparison")
        return

    df = pd.DataFrame(all_results)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    
    ax1.bar(range(len(df)), df['accuracy'], color='skyblue', alpha=0.7)
    ax1.set_xlabel('Experiments')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['experiment'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    
    for i, v in enumerate(df['accuracy']):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    
    ax2.bar(range(len(df)), df['loss'], color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Experiments')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss Comparison', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['experiment'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    
    for i, v in enumerate(df['loss']):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    comparison_path = os.path.join(result_dir, 'plots', 'experiment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_path}")
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        plot_training_curves(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2:
        create_comparison_plot(sys.argv[1])
EOF

echo "Single Domain Training with Cross-Domain Testing"

for shot in 1 5; do
    for source_domain in HuffPost; do
        for path in 01 02 03 04 05; do
            echo "Training on ${source_domain}, testing cross-domain (${shot}-shot, split ${path})..."

            
            exp_dir="${RESULT_BASE_PATH}/single_${source_domain}_${shot}shot_${path}"
            mkdir -p $exp_dir

            
            python src_org/cross_domain_main_single.py \
                --comment "${comment}-single-${source_domain}-${shot}shot" \
                --target_domains $source_domain \
                --dataFile "${DATA_BASE_PATH}/${source_domain}/few_shot/${path}" \
                --fileVocab="${BERT_PATH}" \
                --fileModelConfig="${BERT_PATH}/config.json" \
                --fileModel="${BERT_PATH}" \
                --fileModelSave="${exp_dir}" \
                --numKShot $shot \
                --numQShot 25 \
                --sample 3 \
                --numDevice=$cuda \
                --numFreeze 6 \
                --sampling_strategy "single_domain" \
                --domain_loss_weight 0.1294185347338246 \
                --adversarial_loss_weight 0.036497368493134855 \
                --epochs 107 \
                --episodeTrain 178 \
                --episodeTest 1000 \
                --learning_rate 5.040102673799337e-06 \
                --patience 26 \
                --log_file "${exp_dir}/training_log.json" \
                --save_plots True \
                --dropout_rate 0.3968299882693121 \
                --weight_decay 0.02008162021741798 \
                --gradient_accumulation_steps 3

            
            if [ -f "${exp_dir}/training_log.json" ]; then
                echo "Generating training curves for ${source_domain}_${shot}shot_${path}..."
                python $RESULT_BASE_PATH/plot_training_curves.py \
                    "${exp_dir}/training_log.json" \
                    "${RESULT_BASE_PATH}/plots" \
                    "${source_domain}_${shot}shot_${path}"
            fi

            echo "Completed ${shot}-shot experiment on ${source_domain} with data split ${path}"
            echo "------------------------------------------"
        done
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: $RESULT_BASE_PATH"
echo "Plots saved in: $RESULT_BASE_PATH/plots"
echo "=========================================="


echo "Generating comprehensive statistics for 5 experimental runs..."

python -c "
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob

result_dir = '$RESULT_BASE_PATH'

def extract_metrics_from_results():
    \"\"\"result\"\"\"
    all_results = []

    
    patterns = [
        '**/final_results.json',
        '**/enhanced_final_results.json',
        '**/cross_domain_results.json',
        '**/*_results.json'
    ]

    for pattern in patterns:
        for filepath in glob.glob(os.path.join(result_dir, pattern), recursive=True):
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        continue

                   
                    if filepath.endswith('cross_domain_results.json'):
                        for line in content.split('\n'):
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    result = parse_result_data(data, filepath)
                                    if result:
                                        all_results.append(result)
                                except json.JSONDecodeError:
                                    continue
                    else:
                        
                        try:
                            data = json.loads(content)
                            result = parse_result_data(data, filepath)
                            if result:
                                all_results.append(result)
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f'Error reading {filepath}: {e}')
                continue

    return all_results

def parse_result_data(data, filepath):
    \"\"\"aaa\"\"\"
    result = {}

    
    path_parts = Path(filepath).parts
    exp_name = None
    for part in path_parts:
        if 'single_' in part and '_shot_' in part:
            exp_name = part
            break

    if not exp_name:
        return None

    
    parts = exp_name.split('_')
    if len(parts) >= 4:
        domain = parts[1]
        shot_info = parts[2]  
        split_num = parts[3]  

        shot = shot_info.replace('shot', '')

        result['domain'] = domain
        result['shot'] = int(shot)
        result['split'] = split_num
        result['experiment'] = exp_name

   
    metrics_found = False

    
    if 'test_results' in data:
        test_results = data['test_results']
        result['accuracy'] = float(test_results.get('accuracy', 0))
        result['f1'] = float(test_results.get('f1', 0))
        result['precision'] = float(test_results.get('precision', 0))
        result['recall'] = float(test_results.get('recall', 0))
        result['auc'] = float(test_results.get('auc', 0))
        metrics_found = True

    
    elif 'results' in data:
        results = data['results']
        result['accuracy'] = float(results.get('accuracy', 0))
        result['f1'] = float(results.get('f1', 0))
        result['precision'] = float(results.get('precision', 0))
        result['recall'] = float(results.get('recall', 0))
        result['auc'] = float(results.get('auc', 0))
        metrics_found = True

    
    elif 'accuracy' in data:
        result['accuracy'] = float(data.get('accuracy', 0))
        result['f1'] = float(data.get('f1', 0))
        result['precision'] = float(data.get('precision', 0))
        result['recall'] = float(data.get('recall', 0))
        result['auc'] = float(data.get('auc', 0))
        metrics_found = True

    return result if metrics_found else None

def calculate_statistics():
    \"\"\"bbb\"\"\"
    results = extract_metrics_from_results()

    if not results:
        print('No valid results found!')
        return

    df = pd.DataFrame(results)
    print(f'Found {len(df)} experimental results')
    print(f'Columns: {df.columns.tolist()}')

    if len(df) == 0:
        print('No data to analyze')
        return

    
    required_cols = ['accuracy', 'f1', 'precision', 'recall']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f'Warning: Missing columns: {missing_cols}')
        return

    
    groupby_cols = []
    if 'domain' in df.columns:
        groupby_cols.append('domain')
    if 'shot' in df.columns:
        groupby_cols.append('shot')

    if not groupby_cols:
        print('Cannot group data - missing domain/shot information')
        
        print('\\n' + '='*80)
        print('OVERALL STATISTICS (All Experiments)')
        print('='*80)

        metrics = ['accuracy', 'f1', 'precision', 'recall']
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    print(f'{metric.upper()}:')
                    print(f'  Mean: {values.mean():.4f}')
                    print(f'  Std:  {values.std():.4f}')
                    print(f'  Min:  {values.min():.4f}')
                    print(f'  Max:  {values.max():.4f}')
                    print(f'  Count: {len(values)}')
                    print()
        return

   
    print('\\n' + '='*80)
    print('DETAILED STATISTICS BY EXPERIMENTAL GROUPS')
    print('='*80)

    for group_name, group_df in df.groupby(groupby_cols):
        if isinstance(group_name, tuple):
            group_str = ' - '.join(str(x) for x in group_name)
        else:
            group_str = str(group_name)

        print(f'\\nGroup: {group_str}')
        print('-' * 60)
        print(f'Number of experiments: {len(group_df)}')

        
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
        stats_summary = {}

        for metric in metrics:
            if metric in group_df.columns:
                values = group_df[metric].dropna()
                if len(values) > 0:
                    stats = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    }
                    stats_summary[metric] = stats

                    print(f'{metric.upper()}:')
                    print(f'  Mean ± Std: {stats[\"mean\"]:.4f} ± {stats[\"std\"]:.4f}')
                    print(f'  Range: [{stats[\"min\"]:.4f}, {stats[\"max\"]:.4f}]')
                    print(f'  Count: {stats[\"count\"]}')

        
        print('\\nDetailed Results:')
        for idx, row in group_df.iterrows():
            exp_info = row.get('experiment', f'Exp_{idx}')
            split_info = row.get('split', 'N/A')
            print(f'  {exp_info} (Split {split_info}): Acc={row.get(\"accuracy\", 0):.4f}, F1={row.get(\"f1\", 0):.4f}, P={row.get(\"precision\", 0):.4f}, R={row.get(\"recall\", 0):.4f}')

    
    detailed_path = os.path.join(result_dir, 'detailed_statistics.csv')
    df.to_csv(detailed_path, index=False)
    print(f'\\nDetailed results saved to: {detailed_path}')

    
    if groupby_cols:
        summary_stats = []
        for group_name, group_df in df.groupby(groupby_cols):
            if isinstance(group_name, tuple):
                group_dict = dict(zip(groupby_cols, group_name))
            else:
                group_dict = {groupby_cols[0]: group_name}

            for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
                if metric in group_df.columns:
                    values = group_df[metric].dropna()
                    if len(values) > 0:
                        group_dict.update({
                            f'{metric}_mean': values.mean(),
                            f'{metric}_std': values.std(),
                            f'{metric}_min': values.min(),
                            f'{metric}_max': values.max(),
                            f'{metric}_count': len(values)
                        })

            summary_stats.append(group_dict)

        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_path = os.path.join(result_dir, 'summary_statistics.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f'Summary statistics saved to: {summary_path}')

            
            print('\\n' + '='*80)
            print('SUMMARY TABLE')
            print('='*80)
            print(summary_df.round(4).to_string(index=False))


calculate_statistics()
"


echo "Generating statistical visualization plots..."

python -c "
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

result_dir = '$RESULT_BASE_PATH'


detailed_file = os.path.join(result_dir, 'detailed_statistics.csv')
if os.path.exists(detailed_file):
    df = pd.read_csv(detailed_file)

    if len(df) > 0 and 'accuracy' in df.columns:
        
        plt.style.use('default')
        sns.set_palette('husl')

        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Five-Fold Cross-Validation Results Analysis', fontsize=16, fontweight='bold')

        
        if 'split' in df.columns and len(df['split'].unique()) > 1:
            split_acc = df.groupby('split')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
            axes[0, 0].bar(split_acc['split'], split_acc['mean'],
                          yerr=split_acc['std'], capsize=5, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Accuracy by Data Split', fontweight='bold')
            axes[0, 0].set_xlabel('Data Split')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)

           
            for i, row in split_acc.iterrows():
                axes[0, 0].text(i, row['mean'] + 0.01, f'{row[\"mean\"]:.3f}±{row[\"std\"]:.3f}',
                               ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient split data', ha='center', va='center',
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Accuracy by Data Split', fontweight='bold')

        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in df.columns]

        if available_metrics:
            metric_data = df[available_metrics].melt(var_name='Metric', value_name='Score')
            sns.boxplot(data=metric_data, x='Metric', y='Score', ax=axes[0, 1])
            axes[0, 1].set_title('Metric Distribution', fontweight='bold')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True, alpha=0.3)

        
        if available_metrics:
            means = [df[m].mean() for m in available_metrics]
            stds = [df[m].std() for m in available_metrics]

            x = np.arange(len(available_metrics))
            width = 0.35

            axes[0, 2].bar(x - width/2, means, width, label='Mean', alpha=0.7, color='lightgreen')
            axes[0, 2].bar(x + width/2, stds, width, label='Std Dev', alpha=0.7, color='lightcoral')

            axes[0, 2].set_title('Mean vs Standard Deviation', fontweight='bold')
            axes[0, 2].set_xlabel('Metrics')
            axes[0, 2].set_ylabel('Value')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(available_metrics)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            
            for i, (mean, std) in enumerate(zip(means, stds)):
                axes[0, 2].text(i - width/2, mean + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
                axes[0, 2].text(i + width/2, std + 0.01, f'{std:.3f}', ha='center', va='bottom', fontsize=8)

        
        if 'split' in df.columns and len(df['split'].unique()) >= 5:
            splits = sorted(df['split'].unique())
            split_metrics = df.groupby('split')[available_metrics].mean()

            
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]  

            for split in splits[:5]:  
                if split in split_metrics.index:
                    values = split_metrics.loc[split].tolist()
                    values += values[:1]  

                    axes[1, 0].plot(angles, values, 'o-', linewidth=2,
                                   label=f'Split {split}', alpha=0.7)
                    axes[1, 0].fill(angles, values, alpha=0.15)

            axes[1, 0].set_xticks(angles[:-1])
            axes[1, 0].set_xticklabels(available_metrics)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('Performance Comparison Across Splits', fontweight='bold')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient split data for radar plot',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Performance Comparison Across Splits', fontweight='bold')

       
        convergence_data = []
        for exp_dir in glob.glob(os.path.join(result_dir, 'single_*')):
            log_file = os.path.join(exp_dir, 'training_log.json')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    episodes = []
                    val_accs = []
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                if 'val_accuracy' in data and data['val_accuracy'] > 0:
                                    episodes.append(data.get('episode', 0))
                                    val_accs.append(data['val_accuracy'])
                            except:
                                continue
                    if episodes and val_accs:
                        convergence_data.append((episodes, val_accs))

        if convergence_data:
            for i, (episodes, val_accs) in enumerate(convergence_data[:5]):
                if episodes and val_accs:
                    axes[1, 1].plot(episodes, val_accs, alpha=0.7, label=f'Exp {i+1}')

            axes[1, 1].set_title('Validation Accuracy Convergence', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Validation Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No convergence data available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Validation Accuracy Convergence', fontweight='bold')

        
        axes[1, 2].axis('off')

        
        summary_stats = {}
        for metric in available_metrics:
            values = df[metric].dropna()
            if len(values) > 0:
                summary_stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'count': len(values)
                }

        summary_text = f'''Five-Fold Experiment Statistics:

Total Experiments: {len(df)}

'''

        for metric, stats in summary_stats.items():
            summary_text += f'''{metric.upper()}:
  Mean ± Std: {stats["mean"]:.4f} ± {stats["std"]:.4f}
  Range: [{stats["min"]:.4f}, {stats["max"]:.4f}]
  Experiments: {stats["count"]}

'''

        
        if 'accuracy' in summary_stats and summary_stats['accuracy']['count'] >= 2:
            acc_mean = summary_stats['accuracy']['mean']
            acc_std = summary_stats['accuracy']['std']
            acc_count = summary_stats['accuracy']['count']

            
            import scipy.stats as stats
            confidence_interval = stats.t.interval(
                0.95, acc_count-1,
                loc=acc_mean,
                scale=acc_std/np.sqrt(acc_count)
            )

            summary_text += f'''95% Confidence Interval (Accuracy):
  [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]
'''

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plot_path = os.path.join(result_dir, 'plots', 'five_fold_analysis.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Five-fold analysis plot saved to: {plot_path}')
        plt.close()

        
        print('\\n' + '='*60)
        print('FIVE-FOLD CROSS-VALIDATION SUMMARY')
        print('='*60)

        for metric, stats in summary_stats.items():
            print(f'{metric.upper()}: {stats[\"mean\"]:.4f} ± {stats[\"std\"]:.4f} (n={stats[\"count\"]})')

        if 'accuracy' in summary_stats:
            print(f'\\nBest Accuracy: {summary_stats[\"accuracy\"][\"max\"]:.4f}')
            print(f'Worst Accuracy: {summary_stats[\"accuracy\"][\"min\"]:.4f}')
            print(f'Accuracy Range: {summary_stats[\"accuracy\"][\"max\"] - summary_stats[\"accuracy\"][\"min\"]:.4f}')
else:
    print('No detailed statistics file found')
"

echo "=========================================="
echo "All visualizations completed!"
echo "Check the following directories:"
echo "- Results: $RESULT_BASE_PATH"
echo "- Plots: $RESULT_BASE_PATH/plots"
echo "- Summary: $RESULT_BASE_PATH/experiment_summary.csv"
echo "- Detailed Stats: $RESULT_BASE_PATH/detailed_statistics.csv"
echo "- Summary Stats: $RESULT_BASE_PATH/summary_statistics.csv"
echo "=========================================="
