import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from transformers import pipeline, MarianMTModel, MarianTokenizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

class TextAugmentation:
    def __init__(self, aug_config: Dict = None):
        self.config = aug_config or self._get_default_config()
        self.back_translation_models = {}
        self._init_back_translation()
        
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        ])
    
    def _get_default_config(self) -> Dict:
        return {
            'synonym_replacement': {
                'prob': 0.15, 
                'max_replacements': 3  
            },
            'random_insertion': {
                'prob': 0.15,
                'max_insertions': 2
            },
            'random_deletion': {
                'prob': 0.1,
                'min_length': 3  
            },
            'back_translation': {
                'prob': 0.2,
                'languages': ['de', 'fr']  
            },
            'domain_specific': {
                'news': ['synonym_replacement', 'back_translation'],
                'review': ['synonym_replacement', 'random_deletion'],
                'intent': ['random_insertion', 'random_deletion'],
                'short_text': ['synonym_replacement', 'random_insertion']
            }
        }
    
    def _init_back_translation(self):
        try:
            self.bt_available = True
        except Exception as e:
            print(f"Back translation models not available: {e}")
            self.bt_available = False
    
    def get_domain_type(self, domain_name: str) -> str:
        domain_mapping = {
            'HuffPost': 'news',
            'Amazon': 'review', 
            'Reuters': 'news',
            'Banking77': 'intent',
            'HWU64': 'intent',
            'Liu57': 'intent',
            'Clinc150': 'intent',
            'OOS': 'intent'
        }
        return domain_mapping.get(domain_name, 'short_text')
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and synonym.isalpha():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def synonym_replacement(self, text: str, prob: float = 0.15, max_replacements: int = 3) -> str:
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        
        new_words = words.copy()
        num_replaced = 0
        
        for i, (word, pos) in enumerate(tagged_words):
            if (num_replaced >= max_replacements or 
                random.random() > prob or 
                word.lower() in self.stop_words or
                not word.isalpha()):
                continue
            
            synonyms = self.get_synonyms(word, pos)
            if synonyms:
                new_words[i] = random.choice(synonyms)
                num_replaced += 1
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, prob: float = 0.15, max_insertions: int = 2) -> str:
        words = word_tokenize(text)
        
        for _ in range(max_insertions):
            if random.random() > prob:
                continue
            random_word = random.choice([w for w in words if w.isalpha()])
            synonyms = self.get_synonyms(random_word)
            
            if synonyms:
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_synonym)
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, prob: float = 0.1, min_length: int = 3) -> str:
        words = word_tokenize(text)
        
        if len(words) <= min_length:
            return text
        
        new_words = []
        for word in words:
            if random.random() > prob:
                new_words.append(word)
        
        if len(new_words) < min_length:
            return text
        
        return ' '.join(new_words) if new_words else text
    
    def back_translation(self, text: str, target_lang: str = 'de') -> str:
        if not self.bt_available:
            return text
        
        try:
            model_name_to = f"Helsinki-NLP/opus-mt-en-{target_lang}"
            model_name_back = f"Helsinki-NLP/opus-mt-{target_lang}-en"
            
            if model_name_to not in self.back_translation_models:
                try:
                    tokenizer_to = MarianTokenizer.from_pretrained(model_name_to)
                    model_to = MarianMTModel.from_pretrained(model_name_to)
                    self.back_translation_models[model_name_to] = (tokenizer_to, model_to)
                except:
                    return text
            
            if model_name_back not in self.back_translation_models:
                try:
                    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
                    model_back = MarianMTModel.from_pretrained(model_name_back)
                    self.back_translation_models[model_name_back] = (tokenizer_back, model_back)
                except:
                    return text
            
            tokenizer_to, model_to = self.back_translation_models[model_name_to]
            tokenizer_back, model_back = self.back_translation_models[model_name_back]
            inputs_to = tokenizer_to(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model_to.generate(**inputs_to, max_length=512, num_beams=4, early_stopping=True)
            intermediate_text = tokenizer_to.decode(translated[0], skip_special_tokens=True)
            inputs_back = tokenizer_back(intermediate_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            back_translated = model_back.generate(**inputs_back, max_length=512, num_beams=4, early_stopping=True)
            result = tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
            
            return result if result.strip() else text
            
        except Exception as e:
            print(f"Back translation failed: {e}")
            return text
    
    def domain_adaptive_augment(self, text: str, domain: str, augment_rate: float = 0.3) -> List[str]:
        domain_type = self.get_domain_type(domain)
        available_methods = self.config['domain_specific'].get(domain_type, ['synonym_replacement'])
        
        augmented_texts = []
        
        for method in available_methods:
            if random.random() < augment_rate:
                if method == 'synonym_replacement':
                    config = self.config['synonym_replacement']
                    aug_text = self.synonym_replacement(
                        text, 
                        prob=config['prob'], 
                        max_replacements=config['max_replacements']
                    )
                elif method == 'random_insertion':
                    config = self.config['random_insertion']
                    aug_text = self.random_insertion(
                        text,
                        prob=config['prob'],
                        max_insertions=config['max_insertions']
                    )
                elif method == 'random_deletion':
                    config = self.config['random_deletion']
                    aug_text = self.random_deletion(
                        text,
                        prob=config['prob'],
                        min_length=config['min_length']
                    )
                elif method == 'back_translation':
                    config = self.config['back_translation']
                    if random.random() < config['prob']:
                        target_lang = random.choice(config['languages'])
                        aug_text = self.back_translation(text, target_lang)
                    else:
                        continue
                else:
                    continue
                
                if aug_text and aug_text.strip() != text.strip():
                    augmented_texts.append(aug_text)
        
        return augmented_texts
    
    def progressive_augment(self, texts: List[str], labels: List[str], domains: List[str], 
                          epoch: int, max_epochs: int) -> Tuple[List[str], List[str], List[str]]:
        progress = epoch / max_epochs
        base_rate = 0.1  
        max_rate = 0.5   
        current_rate = base_rate + (max_rate - base_rate) * progress
        
        augmented_texts = []
        augmented_labels = []
        augmented_domains = []
        
        for text, label, domain in zip(texts, labels, domains):
            augmented_texts.append(text)
            augmented_labels.append(label)
            augmented_domains.append(domain)
            if random.random() < current_rate:
                aug_samples = self.domain_adaptive_augment(text, domain, augment_rate=0.5)
                max_aug_per_sample = max(1, int(3 * progress))
                aug_samples = aug_samples[:max_aug_per_sample]
                
                for aug_text in aug_samples:
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
                    augmented_domains.append(domain)
        
        return augmented_texts, augmented_labels, augmented_domains
    
    def class_balanced_augment(self, texts: List[str], labels: List[str], domains: List[str],
                             target_samples_per_class: int = 50) -> Tuple[List[str], List[str], List[str]]:
        from collections import Counter
        label_counts = Counter(labels)
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        augmented_domains = list(domains)
        
        for label, count in label_counts.items():
            if count < target_samples_per_class:
                needed_samples = target_samples_per_class - count
                class_indices = [i for i, l in enumerate(labels) if l == label]
                for _ in range(needed_samples):
                    idx = random.choice(class_indices)
                    original_text = texts[idx]
                    domain = domains[idx]
                    aug_samples = self.domain_adaptive_augment(original_text, domain, augment_rate=0.8)
                    
                    if aug_samples:
                        chosen_aug = random.choice(aug_samples)
                        augmented_texts.append(chosen_aug)
                        augmented_labels.append(label)
                        augmented_domains.append(domain)
        
        return augmented_texts, augmented_labels, augmented_domains

def create_augmenter(domain_name: str = None) -> TextAugmentation:
    config = {
        'synonym_replacement': {'prob': 0.15, 'max_replacements': 3},
        'random_insertion': {'prob': 0.15, 'max_insertions': 2},
        'random_deletion': {'prob': 0.1, 'min_length': 3},
        'back_translation': {'prob': 0.2, 'languages': ['de', 'fr']},
        'domain_specific': {
            'news': ['synonym_replacement', 'back_translation'],
            'review': ['synonym_replacement', 'random_deletion'],
            'intent': ['random_insertion', 'random_deletion'],
            'short_text': ['synonym_replacement', 'random_insertion']
        }
    }
    if domain_name:
        domain_type = TextAugmentation().get_domain_type(domain_name)
        if domain_type == 'intent':
            config['synonym_replacement']['max_replacements'] = 2
            config['random_insertion']['max_insertions'] = 1
            config['random_deletion']['prob'] = 0.05
        elif domain_type == 'news':
            config['synonym_replacement']['max_replacements'] = 4
            config['back_translation']['prob'] = 0.25
    
    return TextAugmentation(config)


def augment_episode_data(support_set: List[Dict], query_set: List[Dict], 
                        augmenter: TextAugmentation, 
                        episode: int = 0, max_episodes: int = 100) -> Tuple[List[Dict], List[Dict]]:
    aug_support_set = []
    
    for sample in support_set:
        text = sample["text"]
        label = sample["label"]
        domain = sample.get("domain", "unknown")

        aug_support_set.append(sample)

        progress = episode / max_episodes if max_episodes > 0 else 0.5
        augment_rate = 0.1 + 0.4 * progress  
        
        if random.random() < augment_rate:

            aug_texts = augmenter.domain_adaptive_augment(text, domain, augment_rate=0.6)

            max_aug = max(1, int(2 * progress))
            aug_texts = aug_texts[:max_aug]
            
            for aug_text in aug_texts:
                aug_sample = sample.copy()
                aug_sample["text"] = aug_text
                aug_support_set.append(aug_sample)
    
    return aug_support_set, query_set
