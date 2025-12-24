import copy
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig, BertModel

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, num_domains, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, x):
        return self.classifier(x)

class TaskAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TaskAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  
        )
        
    def forward(self, x):
        scale = self.adapter(x)
        return x * (1 + scale) 

class CrossDomainBertEncoder(nn.Module):    
    def __init__(self, args):
        super(CrossDomainBertEncoder, self).__init__()
        
        self.args = args
        if hasattr(args, 'numDevice') and torch.cuda.is_available():
            self.device = torch.device('cuda', args.numDevice)
        else:
            self.device = torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_lower_case=True)
        config = BertConfig.from_json_file(args.fileModelConfig)   
        self.bert = BertModel.from_pretrained(args.fileModel, config=config)
        self.bert = self.bert.to(self.device)
        self.agnostic_layers_num = getattr(args, 'agnostic_layers_num', 6)  
        self.specific_layers_num = 12 - self.agnostic_layers_num  
        self.domain_agnostic_layers = nn.ModuleList([
            copy.deepcopy(self.bert.encoder.layer[i]) for i in range(self.agnostic_layers_num)
        ])
        
        self.domain_specific_layers = nn.ModuleDict()
        self.domains = ['BANKING77', 'HWU64','Liu','OOS'] 
        
        for domain in self.domains:
            self.domain_specific_layers[domain] = nn.ModuleList([
                copy.deepcopy(self.bert.encoder.layer[i]) for i in range(self.agnostic_layers_num, 12)
            ])
        
        self.task_adapters = nn.ModuleDict()
        for domain in self.domains:
            self.task_adapters[domain] = TaskAdapter(768)
        
        self.domain_discriminator = DomainDiscriminator(768, len(self.domains))
        
        self.attention = copy.deepcopy(self.bert.encoder.layer[11])
        self.lin = nn.Linear(768, 768)
        self.drop = nn.Dropout(0.1)
        self.la = args.la
        
        self.domain_agnostic_layers = self.domain_agnostic_layers.to(self.device)
        self.domain_specific_layers = self.domain_specific_layers.to(self.device)
        self.task_adapters = self.task_adapters.to(self.device)
        self.domain_discriminator = self.domain_discriminator.to(self.device)
        self.attention = self.attention.to(self.device)
        self.lin = self.lin.to(self.device)
        self.drop = self.drop.to(self.device)
        
        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)

    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def get_domain_from_dataset(self, text):
        if hasattr(self.args, 'dataFile') and self.args.dataFile:
            datafile = self.args.dataFile.lower()
            domain_keywords = {
                '20news': '20News',
                'huffpost': 'HuffPost', 
                'amazon': 'Amazon',
                'reuters': 'Reuters',
                'banking77': 'BANKING77',
                'hwu64': 'HWU64',
                'oos': 'OOS',
                'liu': 'Liu'
            }
        
            for keyword, domain in domain_keywords.items():
                if keyword in datafile:
                    return domain
    
    def forward(self, text, task_classes, domain_name=None):
        sentence = [x for x in text]
        tokenizer = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=128, 
            return_tensors='pt'
        )
        input_ids = tokenizer['input_ids'].to(self.device)
        token_type_ids = tokenizer['token_type_ids'].to(self.device)
        attention_mask = tokenizer['attention_mask'].to(self.device)
        embeddings = self.bert.embeddings(input_ids, token_type_ids)

        hidden_states = embeddings
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape, self.device
        )
        
        for layer in self.domain_agnostic_layers:
            layer_outputs = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
        
        domain_agnostic_features = hidden_states
        if domain_name is None:
            domain_name = self.get_domain_from_dataset(text)
        if not hasattr(self, '_domain_usage_count'):
            self._domain_usage_count = {}
            self._total_calls = 0
    
        if domain_name not in self._domain_usage_count:
            self._domain_usage_count[domain_name] = 0
        self._domain_usage_count[domain_name] += 1
        self._total_calls += 1

        if self._total_calls % 50 == 0:
            print(f"Domain usage after {self._total_calls} calls: {self._domain_usage_count}")
        
        if domain_name in self.domain_specific_layers:
            for layer in self.domain_specific_layers[domain_name]:
                layer_outputs = layer(hidden_states, extended_attention_mask)
                hidden_states = layer_outputs[0]
        
        if domain_name in self.task_adapters:
            adapted_features = self.task_adapters[domain_name](hidden_states.mean(dim=1))
        else:
            adapted_features = hidden_states.mean(dim=1)
        
        if self.la == 1 and task_classes is not None:
            task_labels_inputs = self.tokenizer.batch_encode_plus(
                task_classes, return_tensors='pt', padding=True, truncation=True, max_length=32)
            task_labels_inputs_input_ids = task_labels_inputs['input_ids'].to(self.device)
            task_labels_inputs_token_type_ids = task_labels_inputs['token_type_ids'].to(self.device)
            task_labels_inputs_attention_mask = task_labels_inputs['attention_mask'].to(self.device)

            label_embedding = self.bert(
                task_labels_inputs_input_ids,
                task_labels_inputs_attention_mask,
                task_labels_inputs_token_type_ids
            ).last_hidden_state.mean(dim=1)
            
           
            sentence_embeddings = hidden_states
            label_embeddings = label_embedding.unsqueeze(dim=0).repeat_interleave(
                sentence_embeddings.shape[0], dim=0)
            connect_embeddings = torch.cat((sentence_embeddings, label_embeddings), dim=1)
            outputs = self.lin(connect_embeddings)
            outputs = self.drop(outputs)
            outputs = self.attention(outputs)[0]
            outputs = 0.1 * outputs + 0.9 * connect_embeddings
            adapted_features = outputs[:, 0, :]
        return {
            'final_features': adapted_features,
            'domain_agnostic_features': domain_agnostic_features.mean(dim=1),
            'domain_name': domain_name
        }

class CrossDomainSampler(nn.Module):
    def __init__(self, args):
        super(CrossDomainSampler, self).__init__()
        self.nway = args.numNWay
        self.kshot = args.numKShot
        self.qshot = args.numQShot
        self.dim = 768
        self.k = args.k
        self.num_sampled = args.sample

    def forward(self, support_embeddings, query_embeddings):
        similarity = euclidean_dist(support_embeddings, query_embeddings)
        similarity = similarity.view(self.nway, 1, -1)
        similarity = similarity.view(self.nway, self.kshot, -1)
      
        values, indices = similarity.topk(self.k, dim=2, largest=False, sorted=True)  
        
        acc = []
        for i in range(self.nway):
            min_index = i * self.qshot
            max_index = (i+1) * self.qshot - 1
            for j in range(self.kshot):
                count = 0.0
                for z in range(self.k):
                    if indices[i][j][z] >= min_index and indices[i][j][z] <= max_index:
                        count += 1
                acc.append(count/(self.k + 0.0)) 
       
        acc = torch.tensor(acc).mean()
        nindices = indices.view(-1, self.k)
       
        convex_feat = []
        for i in range(nindices.shape[0]):
            convex_feat.append(query_embeddings.index_select(0, nindices[i]))
        convex_feat = torch.stack(convex_feat)
        sampled_data = convex_feat.view(self.nway, self.kshot*self.k, self.dim)
       
        return sampled_data, acc

class CrossDomainMetaModel(nn.Module):
    def __init__(self, args):
        super(CrossDomainMetaModel, self).__init__()
        self.args = args
        self.encoder = CrossDomainBertEncoder(args)
        self.sampler = CrossDomainSampler(args)
        self.domain_loss_weight = 0.1
        
    def forward(self, text, label, domain_name=None):
        support_size = self.args.numNWay * self.args.numKShot
        query_size = self.args.numNWay * self.args.numQShot
        encoded_outputs = self.encoder(text, label, domain_name)
        text_embedding = encoded_outputs['final_features']
        domain_agnostic_features = encoded_outputs['domain_agnostic_features']
        actual_domain = encoded_outputs['domain_name'] 
        support_embeddings = text_embedding[:support_size]
        query_embeddings = text_embedding[support_size:]
        c_prototypes = support_embeddings.view(self.args.numNWay, -1, support_embeddings.shape[1])
        original_prototypes = c_prototypes.mean(dim=1)
        sampled_data, acc = self.sampler(support_embeddings, query_embeddings)
        prototypes = torch.cat((c_prototypes, sampled_data), dim=1)
        prototypes = torch.mean(prototypes, dim=1)
        
        return {
            'prototypes': prototypes,
            'query_embeddings': query_embeddings,
            'acc': acc,
            'original_prototypes': original_prototypes,
            'sampled_data': sampled_data,
            'domain_agnostic_features': domain_agnostic_features,
            'domain_name': actual_domain
        }
