import csv
import os
import torch
import json
import numpy as np
from tqdm import tqdm
import random
import argparse
import json
import os
import warnings
from datetime import datetime
from cross_domain_encoder import CrossDomainMetaModel
from cross_domain_losses import CrossDomainLoss
from cross_domain_data_loader import init_cross_domain_dataloader, get_cross_domain_label_dict

from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*device.*deprecated.*", category=FutureWarning)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log_training_progress(log_file, episode, train_loss, train_acc, val_loss, val_acc, 
                         domain_loss=0.0, adversarial_loss=0.0):
    if log_file is None:
        return
    def safe_float(value):
        if hasattr(value, 'item'): 
            return float(value.item())
        else:
            return float(value)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'episode': int(episode),
        'train_loss': safe_float(train_loss),
        'train_accuracy': safe_float(train_acc),
        'val_loss': safe_float(val_loss),
        'val_accuracy': safe_float(val_acc),
        'domain_loss': safe_float(domain_loss),
        'adversarial_loss': safe_float(adversarial_loss)
    }
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Warning: Failed to write log entry: {e}")
        
def get_cross_domain_parser():
    parser = argparse.ArgumentParser(description='Cross-Domain Meta-Learning for Few-Shot Text Classification')
    parser.add_argument('--comment', type=str, help='experiment comment', default='CrossDomainMeta')
    parser.add_argument('--dataset', type=str, help='dataset name', default='multi_domain')
    parser.add_argument('--dataFile', type=str, help='path to dataset', required=True)
    parser.add_argument('--fileVocab', type=str, help='path to pretrained model vocab', required=True)
    parser.add_argument('--fileModelConfig', type=str, help='path to pretrained model config', required=True)
    parser.add_argument('--fileModel', type=str, help='path to pretrained model', required=True)
    parser.add_argument('--fileModelSave', type=str, help='path to save model', required=True)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--numNWay', type=int, default=5, help='number of classes per episode')
    parser.add_argument('--numKShot', type=int, default=5, help='number of instances per class')
    parser.add_argument('--numQShot', type=int, default=25, help='number of queries per class')
    parser.add_argument('--episodeTrain', type=int, default=100, help='number of tasks per epoch in training')
    parser.add_argument('--episodeTest', type=int, default=1000, help='number of tasks per epoch in testing')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--numFreeze', type=int, default=6, help='number of frozen layers')
    parser.add_argument('--numDevice', type=int, default=0, help='gpu device id')
    parser.add_argument('--warmup_steps', type=int, default=189, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay ratio')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='Number of gradient accumulation steps')
    parser.add_argument('--k', type=int, default=15, help='top k for sampling')
    parser.add_argument('--sample', type=int, default=3, help='number of generated samples per shot')
    parser.add_argument('--la', type=int, default=1, help='label adapter flag')
    parser.add_argument('--T', type=int, default=5, help='temperature for contrastive loss')
    parser.add_argument('--agnostic_layers_num', type=int, default=6, help='number of domain-agnostic layers')
    parser.add_argument('--target_domains', nargs='+', default=['HuffPost', 'Amazon','20News','Reuters','BANKING77','OOS','HWU64','Liu'],
                        help='target domains for cross-domain learning')
    parser.add_argument('--domain_loss_weight', type=float, default=0.1,
                        help='weight for domain adversarial loss')
    parser.add_argument('--adversarial_loss_weight', type=float, default=0.01,
                        help='weight for adversarial training')
    parser.add_argument('--sampling_strategy', type=str, default='mixed',
                        choices=['mixed', 'single_domain', 'cross_domain'],
                        help='sampling strategy for episodes')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='number of adaptation steps for meta-learning')
    parser.add_argument('--log_file', type=str, default=None, 
                    help='Path to save training log in JSON format')
    parser.add_argument('--save_plots', type=bool, default=False, 
                    help='Whether to save training plots')
    return parser


def init_cross_domain_model(args):
    if torch.cuda.is_available() and hasattr(args, 'numDevice'):
        device = torch.device('cuda', args.numDevice)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    model = CrossDomainMetaModel(args)
    model = model.to(device)
    return model


def init_cross_domain_optim(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


def init_lr_scheduler(args, optim):
    t_total = args.epochs * args.episodeTrain
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    return scheduler


def deal_cross_domain_data(support_set, query_set, episode_labels):
    text, labels, domains = [], [], []

    for x in support_set:
        text.append(x["text"])
        labels.append(x["label"])
        domains.append(x.get("domain", "unknown"))

    for x in query_set:
        text.append(x["text"])
        labels.append(x["label"])
        domains.append(x.get("domain", "unknown"))
    label_ids = []
    for label in labels:
        tmp = []
        for l in episode_labels:
            if l == label:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)

    return text, label_ids, domains


def cross_domain_train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    print("Starting Cross-Domain Meta-Learning Training...")
    loss_fn = CrossDomainLoss(args)
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_acc = 0
    cycle = 0

    acc_best_model_path = os.path.join(args.fileModelSave, 'cross_domain_best_model.pth')
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, 'w') as f:
            pass
        print(f"Training log will be saved to: {args.log_file}")

    for epoch in range(args.epochs):
        print(f'=== Cross-Domain Epoch: {epoch} ===')
        model.train()

        if cycle == args.patience:
            print("Early stopping triggered")
            break

        epoch_losses = []
        epoch_metrics = defaultdict(list)
        episode_count = 0

        for i, batch in tqdm(enumerate(tr_dataloader), desc="Training"):
            optim.zero_grad()

            if len(batch) == 4:  
                support_set, query_set, episode_labels, episode_domain = batch
            else:  
                support_set, query_set, episode_labels = batch
                episode_domain = None

            
            text, labels, domains = deal_cross_domain_data(support_set, query_set, episode_labels)
            if episode_domain and episode_domain in get_cross_domain_label_dict(episode_domain):
                label_dict = get_cross_domain_label_dict(episode_domain)
                id2label = {v: k for k, v in label_dict.items()}
                label_text = [id2label.get(i, f"class_{i}") for i in range(len(episode_labels))]
            else:
                label_text = episode_labels

            model_outputs = model(text, label_text, episode_domain)

            loss, p, r, f, acc, auc, topk_acc = loss_fn(
                model_outputs, labels, domains
            )

            loss.backward()
            optim.step()
            lr_scheduler.step()

            epoch_losses.append(loss.item())
            epoch_metrics['precision'].append(p)
            epoch_metrics['recall'].append(r)
            epoch_metrics['f1'].append(f)
            epoch_metrics['accuracy'].append(acc)
            epoch_metrics['auc'].append(auc)
            epoch_metrics['topk_acc'].append(topk_acc)
            
            episode_count += 1
            current_episode = epoch * args.episodeTrain + episode_count
            
            if episode_count % 10 == 0 or i == len(tr_dataloader) - 1:
                recent_train_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                recent_train_acc = np.mean(epoch_metrics['accuracy'][-10:]) if len(epoch_metrics['accuracy']) >= 10 else np.mean(epoch_metrics['accuracy'])
                
                val_loss_current = 0.0
                val_acc_current = 0.0
                
                if val_dataloader is not None and (episode_count % 20 == 0 or i == len(tr_dataloader) - 1):
                    model.eval()
                    val_losses_temp = []
                    val_accs_temp = []
                    
                    with torch.no_grad():
                       
                        val_batch_count = 0
                        for val_batch in val_dataloader:
                            if val_batch_count >= 10:  
                                break
                                
                            if len(val_batch) == 4:
                                val_support_set, val_query_set, val_episode_labels, val_episode_domain = val_batch
                            else:
                                val_support_set, val_query_set, val_episode_labels = val_batch
                                val_episode_domain = None

                            val_text, val_labels, val_domains = deal_cross_domain_data(val_support_set, val_query_set, val_episode_labels)

                            if val_episode_domain and val_episode_domain in get_cross_domain_label_dict(val_episode_domain):
                                val_label_dict = get_cross_domain_label_dict(val_episode_domain)
                                val_id2label = {v: k for k, v in val_label_dict.items()}
                                val_label_text = [val_id2label.get(i, f"class_{i}") for i in range(len(val_episode_labels))]
                            else:
                                val_label_text = val_episode_labels

                            val_model_outputs = model(val_text, val_label_text, val_episode_domain)
                            val_loss, val_p, val_r, val_f, val_acc, val_auc, val_topk_acc = loss_fn(
                                val_model_outputs, val_labels, val_domains
                            )
                            
                            val_losses_temp.append(val_loss.item())
                            val_accs_temp.append(val_acc)
                            val_batch_count += 1
                    
                    model.train()
                    val_loss_current = np.mean(val_losses_temp) if val_losses_temp else 0.0
                    val_acc_current = np.mean(val_accs_temp) if val_accs_temp else 0.0
                
                log_training_progress(
                    args.log_file,
                    current_episode,
                    float(recent_train_loss),  
                    float(recent_train_acc),   
                    float(val_loss_current),   
                    float(val_acc_current),    
                    domain_loss=getattr(loss_fn, 'last_domain_loss', 0.0),  
                    adversarial_loss=getattr(loss_fn, 'last_adversarial_loss', 0.0)  
                )

            if i % 20 == 0:  
                print(f'Batch {i}: Loss={loss.item():.4f}, Acc={acc:.4f}, F1={f:.4f}')

       
        avg_loss = float(np.mean(epoch_losses)) 
        avg_metrics = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        print(f'Epoch {epoch} Training Results:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {avg_metrics["accuracy"]:.4f}')
        print(f'  F1: {avg_metrics["f1"]:.4f}')
        print(f'  Precision: {avg_metrics["precision"]:.4f}')
        print(f'  Recall: {avg_metrics["recall"]:.4f}')

        
        train_metrics['loss'].append(avg_loss)
        for k, v in avg_metrics.items():
            train_metrics[k].append(v)

        
        val_loss = 0.0
        val_avg_metrics = {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        if val_dataloader is not None:
            val_loss, val_avg_metrics = cross_domain_evaluate(
                args, val_dataloader, model, loss_fn, "Validation"
            )

            
            val_metrics['loss'].append(val_loss)
            for k, v in val_avg_metrics.items():
                val_metrics[k].append(v)

            
            cycle += 1
            if val_avg_metrics['accuracy'] >= best_acc:
                torch.save(model.state_dict(), acc_best_model_path)
                best_acc = val_avg_metrics['accuracy']
                cycle = 0
                print(f'  New best accuracy: {best_acc:.4f}')
        
        
        epoch_episode_num = (epoch + 1) * args.episodeTrain
        log_training_progress(
            args.log_file,
            epoch_episode_num,
            avg_loss,
            avg_metrics['accuracy'],
            val_loss,
            val_avg_metrics['accuracy'],
            domain_loss=0.0,  
            adversarial_loss=0.0  
        )

    
    save_training_history(args, train_metrics, val_metrics)

    return model


def cross_domain_evaluate(args, dataloader, model, loss_fn, phase_name="Test"):
    print(f"Starting {phase_name}...")

    model.eval()
    epoch_losses = []
    epoch_metrics = defaultdict(list)
    domain_results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=phase_name):
            if len(batch) == 4:
                support_set, query_set, episode_labels, episode_domain = batch
            else:
                support_set, query_set, episode_labels = batch
                episode_domain = None

            text, labels, domains = deal_cross_domain_data(support_set, query_set, episode_labels)

            if episode_domain and episode_domain in get_cross_domain_label_dict(episode_domain):
                label_dict = get_cross_domain_label_dict(episode_domain)
                id2label = {v: k for k, v in label_dict.items()}
                label_text = [id2label.get(i, f"class_{i}") for i in range(len(episode_labels))]
            else:
                label_text = episode_labels

            model_outputs = model(text, label_text, episode_domain)

            loss, p, r, f, acc, auc, topk_acc = loss_fn(
                model_outputs, labels, domains
            )

            epoch_losses.append(loss.item())
            epoch_metrics['precision'].append(p)
            epoch_metrics['recall'].append(r)
            epoch_metrics['f1'].append(f)
            epoch_metrics['accuracy'].append(acc)
            epoch_metrics['auc'].append(auc)
            epoch_metrics['topk_acc'].append(topk_acc)

            if episode_domain:
                domain_results[episode_domain]['precision'].append(p)
                domain_results[episode_domain]['recall'].append(r)
                domain_results[episode_domain]['f1'].append(f)
                domain_results[episode_domain]['accuracy'].append(acc)
                domain_results[episode_domain]['auc'].append(auc)

    avg_loss = np.mean(epoch_losses)
    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

    print(f'{phase_name} Results:')
    print(f'  Overall Loss: {avg_loss:.4f}')
    print(f'  Overall Accuracy: {avg_metrics["accuracy"]:.4f}')
    print(f'  Overall F1: {avg_metrics["f1"]:.4f}')

    for domain, metrics in domain_results.items():
        domain_avg = {k: np.mean(v) for k, v in metrics.items()}
        print(f'  {domain} - Acc: {domain_avg["accuracy"]:.4f}, F1: {domain_avg["f1"]:.4f}')

    return avg_loss, avg_metrics


def save_training_history(args, train_metrics, val_metrics):
    history_path = os.path.join(args.fileModelSave, 'training_history.json')
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  
        else:
            return obj

    history = {
        'train': convert_to_serializable({k: v for k, v in train_metrics.items()}),
        'val': convert_to_serializable({k: v for k, v in val_metrics.items()})
    }

    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")


def save_results(args, results):
    csv_path = os.path.join(args.fileModelSave, 'cross_domain_results.csv')
    with open(csv_path, 'a+', newline="") as f:
        writer = csv.writer(f)
        data = [
            "comment", args.comment,
            "domains", "-".join(args.target_domains),
            "shot", args.numKShot,
            "accuracy", float(results['accuracy']),  
            "f1", float(results['f1']),
            "precision", float(results['precision']),
            "recall", float(results['recall'])
        ]
        writer.writerow(data)
    json_path = os.path.join(args.fileModelSave, 'cross_domain_results.json')
    result_data = {
        "comment": args.comment,
        "domains": args.target_domains,
        "shot": args.numKShot,
        "sampling_strategy": args.sampling_strategy,
        "results": {
            "accuracy": float(results['accuracy']),
            "f1": float(results['f1']),
            "precision": float(results['precision']),
            "recall": float(results['recall']),
            "auc": float(results.get('auc', 0.0)),
            "topk_acc": float(results.get('topk_acc', 0.0))
        }
    }

    try:
        with open(json_path, 'a+') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving JSON results: {e}")


def write_args_to_json(args):
    config_path = os.path.join(args.fileModelSave, 'cross_domain_config.json')
    args_dict = vars(args)

    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    print(f"Configuration saved to {config_path}")

def save_final_results(save_path, test_results, args):
    def ensure_serializable(obj):
        if isinstance(obj, dict):
            return {k: ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
            return float(obj) if np.isscalar(obj) else obj.tolist()
        elif hasattr(obj, 'item'):  
            return obj.item()
        else:
            return obj
    
    results = {
        'experiment_config': {
            'comment': args.comment,
            'target_domains': args.target_domains,
            'numKShot': args.numKShot,
            'numQShot': args.numQShot,
            'sampling_strategy': args.sampling_strategy,
            'domain_loss_weight': float(args.domain_loss_weight),
            'adversarial_loss_weight': float(args.adversarial_loss_weight),
            'learning_rate': float(args.learning_rate),
            'epochs': args.epochs,
            'episodeTrain': args.episodeTrain,
            'episodeTest': args.episodeTest
        },
        'test_results': ensure_serializable(test_results),
        'timestamp': datetime.now().isoformat()
    }
    result_file = os.path.join(save_path, 'final_results.json')
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {result_file}")
    except Exception as e:
        print(f"Error saving final results: {e}")


def main():
    parser = get_cross_domain_parser()
    args = parser.parse_args()

    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)
    
    if args.log_file is None:
        args.log_file = os.path.join(args.fileModelSave, 'training_log.json')

    write_args_to_json(args)

    set_seed(args.seed)

    print("=" * 50)
    print("Cross-Domain Meta-Learning for Few-Shot Text Classification")
    print("=" * 50)
    print(f"Target domains: {args.target_domains}")
    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"N-way K-shot: {args.numNWay}-way {args.numKShot}-shot")
    print(f"Log file: {args.log_file}")  
    print("=" * 50)

    model = init_cross_domain_model(args)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    try:
        tr_dataloader, tr_dataset = init_cross_domain_dataloader(args, 'train', args.target_domains)
        val_dataloader, val_dataset = init_cross_domain_dataloader(args, 'valid', args.target_domains)
        test_dataloader, test_dataset = init_cross_domain_dataloader(args, 'test', args.target_domains)

        print(f"Datasets loaded:")
        for domain in args.target_domains:
            if domain in tr_dataset.domain_datasets:
                print(f"  {domain}: {len(tr_dataset.domain_datasets[domain])} samples")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    optimizer = init_cross_domain_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optimizer)
    try:
        model = cross_domain_train(
            args=args,
            tr_dataloader=tr_dataloader,
            model=model,
            optim=optimizer,
            lr_scheduler=lr_scheduler,
            val_dataloader=val_dataloader
        )
        print("Training completed successfully!")

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    try:
        best_model_path = os.path.join(args.fileModelSave, 'cross_domain_best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print("Best model loaded for testing")
        loss_fn = CrossDomainLoss(args)
        test_loss, test_metrics = cross_domain_evaluate(
            args, test_dataloader, model, loss_fn, "Final Test"
        )
        save_final_results(args.fileModelSave, test_metrics, args)
        save_results(args, test_metrics)

        print("=" * 50)
        print("FINAL RESULTS:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"Testing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
