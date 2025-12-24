import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from cross_domain_encoder import CrossDomainMetaModel
from cross_domain_losses import CrossDomainLoss
from cross_domain_data_loader import init_cross_domain_dataloader, get_cross_domain_label_dict

def cross_domain_test_single_target(args):
    model = CrossDomainMetaModel(args)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    print(f"Testing model trained on {args.source_domain} on target domain {args.target_domain}")
    
    test_dataloader, test_dataset = init_cross_domain_dataloader(args, 'test', [args.target_domain])
    
    loss_fn = CrossDomainLoss(args)
    
    test_results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Testing on {args.target_domain}"):
            if len(batch) == 4:
                support_set, query_set, episode_labels, episode_domain = batch
            else:
                support_set, query_set, episode_labels = batch
                episode_domain = args.target_domain
            
            text, labels, domains = deal_cross_domain_data(support_set, query_set, episode_labels)
            
            if episode_domain in get_cross_domain_label_dict(episode_domain):
                label_dict = get_cross_domain_label_dict(episode_domain)
                id2label = {v: k for k, v in label_dict.items()}
                label_text = [id2label.get(i, f"class_{i}") for i in range(len(episode_labels))]
            else:
                label_text = episode_labels
            
            model_outputs = model(text, label_text, episode_domain)
            
            loss, p, r, f, acc, auc, topk_acc = loss_fn(model_outputs, labels, domains)
            
            test_results.append({
                'loss': loss.item(),
                'precision': p,
                'recall': r,
                'f1': f,
                'accuracy': acc,
                'auc': auc,
                'topk_acc': topk_acc
            })
    
    avg_results = {}
    for key in test_results[0].keys():
        avg_results[key] = float(np.mean([result[key] for result in test_results]))
    
    result_data = {
        'source_domain': args.source_domain,
        'target_domain': args.target_domain,
        'shot': args.numKShot,
        'results': avg_results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Cross-domain test results:")
    print(f"  Source: {args.source_domain} → Target: {args.target_domain}")
    print(f"  Accuracy: {avg_results['accuracy']:.4f}")
    print(f"  F1: {avg_results['f1']:.4f}")
    print(f"  Results saved to: {args.output_file}")
    
    return avg_results

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, required=True)
    parser.add_argument('--target_domain', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataFile', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--numKShot', type=int, default=5)
    parser.add_argument('--numQShot', type=int, default=25)
    parser.add_argument('--numNWay', type=int, default=5)
    parser.add_argument('--numDevice', type=int, default=0)
    parser.add_argument('--episodeTest', type=int, default=1000)
    parser.add_argument('--fileVocab', type=str, default='./models/bert-base-uncased')
    parser.add_argument('--fileModelConfig', type=str, default='./models/bert-base-uncased/config.json')
    parser.add_argument('--fileModel', type=str, default='./models/bert-base-uncased')
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--numFreeze', type=int, default=4)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--domain_loss_weight', type=float, default=0.1)
    parser.add_argument('--adversarial_loss_weight', type=float, default=0.01)
    
    args = parser.parse_args()
    cross_domain_test_single_target(args)
