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

# 确保下载必要的NLTK数据
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
    """
    针对跨域元学习的文本数据增强类
    支持多种增强策略，适配不同类型的文本分类数据集
    """
    
    def __init__(self, aug_config: Dict = None):
        """
        初始化数据增强器
        
        Args:
            aug_config: 增强配置字典
        """
        self.config = aug_config or self._get_default_config()
        self.back_translation_models = {}
        self._init_back_translation()
        
        # 停用词列表（避免替换重要词汇）
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        ])
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'synonym_replacement': {
                'prob': 0.15,  # 替换概率
                'max_replacements': 3  # 最大替换数量
            },
            'random_insertion': {
                'prob': 0.15,
                'max_insertions': 2
            },
            'random_deletion': {
                'prob': 0.1,
                'min_length': 3  # 保持最小长度
            },
            'back_translation': {
                'prob': 0.2,
                'languages': ['de', 'fr']  # 德语、法语
            },
            'domain_specific': {
                'news': ['synonym_replacement', 'back_translation'],
                'review': ['synonym_replacement', 'random_deletion'],
                'intent': ['random_insertion', 'random_deletion'],
                'short_text': ['synonym_replacement', 'random_insertion']
            }
        }
    
    def _init_back_translation(self):
        """初始化回译模型"""
        try:
            # 只在需要时加载模型
            self.bt_available = True
        except Exception as e:
            print(f"Back translation models not available: {e}")
            self.bt_available = False
    
    def get_domain_type(self, domain_name: str) -> str:
        """根据域名确定域类型"""
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
        """获取词汇的同义词"""
        synonyms = set()
        
        # 获取WordNet同义词
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and synonym.isalpha():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def synonym_replacement(self, text: str, prob: float = 0.15, max_replacements: int = 3) -> str:
        """同义词替换增强"""
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
        """随机插入增强"""
        words = word_tokenize(text)
        
        for _ in range(max_insertions):
            if random.random() > prob:
                continue
                
            # 随机选择一个词获取其同义词
            random_word = random.choice([w for w in words if w.isalpha()])
            synonyms = self.get_synonyms(random_word)
            
            if synonyms:
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_synonym)
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, prob: float = 0.1, min_length: int = 3) -> str:
        """随机删除增强"""
        words = word_tokenize(text)
        
        if len(words) <= min_length:
            return text
        
        new_words = []
        for word in words:
            if random.random() > prob:
                new_words.append(word)
        
        # 确保不会删除太多词
        if len(new_words) < min_length:
            return text
        
        return ' '.join(new_words) if new_words else text
    
    def back_translation(self, text: str, target_lang: str = 'de') -> str:
        """回译增强"""
        if not self.bt_available:
            return text
        
        try:
            # 使用Helsinki-NLP的MarianMT模型进行回译
            model_name_to = f"Helsinki-NLP/opus-mt-en-{target_lang}"
            model_name_back = f"Helsinki-NLP/opus-mt-{target_lang}-en"
            
            # 翻译到目标语言
            if model_name_to not in self.back_translation_models:
                try:
                    tokenizer_to = MarianTokenizer.from_pretrained(model_name_to)
                    model_to = MarianMTModel.from_pretrained(model_name_to)
                    self.back_translation_models[model_name_to] = (tokenizer_to, model_to)
                except:
                    return text
            
            # 翻译回英语
            if model_name_back not in self.back_translation_models:
                try:
                    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
                    model_back = MarianMTModel.from_pretrained(model_name_back)
                    self.back_translation_models[model_name_back] = (tokenizer_back, model_back)
                except:
                    return text
            
            # 执行翻译
            tokenizer_to, model_to = self.back_translation_models[model_name_to]
            tokenizer_back, model_back = self.back_translation_models[model_name_back]
            
            # 英语 -> 目标语言
            inputs_to = tokenizer_to(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model_to.generate(**inputs_to, max_length=512, num_beams=4, early_stopping=True)
            intermediate_text = tokenizer_to.decode(translated[0], skip_special_tokens=True)
            
            # 目标语言 -> 英语
            inputs_back = tokenizer_back(intermediate_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            back_translated = model_back.generate(**inputs_back, max_length=512, num_beams=4, early_stopping=True)
            result = tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
            
            return result if result.strip() else text
            
        except Exception as e:
            print(f"Back translation failed: {e}")
            return text
    
    def domain_adaptive_augment(self, text: str, domain: str, augment_rate: float = 0.3) -> List[str]:
        """
        根据域特性进行自适应数据增强
        
        Args:
            text: 原始文本
            domain: 域名
            augment_rate: 增强比例
            
        Returns:
            增强后的文本列表
        """
        domain_type = self.get_domain_type(domain)
        available_methods = self.config['domain_specific'].get(domain_type, ['synonym_replacement'])
        
        augmented_texts = []
        
        # 根据域类型选择合适的增强方法
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
                
                # 确保增强文本有效且不同于原文本
                if aug_text and aug_text.strip() != text.strip():
                    augmented_texts.append(aug_text)
        
        return augmented_texts
    
    def progressive_augment(self, texts: List[str], labels: List[str], domains: List[str], 
                          epoch: int, max_epochs: int) -> Tuple[List[str], List[str], List[str]]:
        """
        渐进式数据增强：训练初期增强较少，后期增强较多
        
        Args:
            texts: 文本列表
            labels: 标签列表  
            domains: 域列表
            epoch: 当前epoch
            max_epochs: 总epoch数
            
        Returns:
            增强后的(texts, labels, domains)
        """
        # 计算渐进式增强率
        progress = epoch / max_epochs
        base_rate = 0.1  # 初始增强率
        max_rate = 0.5   # 最大增强率
        current_rate = base_rate + (max_rate - base_rate) * progress
        
        augmented_texts = []
        augmented_labels = []
        augmented_domains = []
        
        for text, label, domain in zip(texts, labels, domains):
            # 保留原始数据
            augmented_texts.append(text)
            augmented_labels.append(label)
            augmented_domains.append(domain)
            
            # 根据当前增强率决定是否增强
            if random.random() < current_rate:
                aug_samples = self.domain_adaptive_augment(text, domain, augment_rate=0.5)
                
                # 限制每个样本的增强数量，避免过度增强
                max_aug_per_sample = max(1, int(3 * progress))
                aug_samples = aug_samples[:max_aug_per_sample]
                
                for aug_text in aug_samples:
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
                    augmented_domains.append(domain)
        
        return augmented_texts, augmented_labels, augmented_domains
    
    def class_balanced_augment(self, texts: List[str], labels: List[str], domains: List[str],
                             target_samples_per_class: int = 50) -> Tuple[List[str], List[str], List[str]]:
        """
        类别平衡增强：为样本较少的类别生成更多增强数据
        
        Args:
            texts: 文本列表
            labels: 标签列表
            domains: 域列表
            target_samples_per_class: 目标每类样本数
            
        Returns:
            平衡后的(texts, labels, domains)
        """
        from collections import Counter
        
        # 统计每个类别的样本数
        label_counts = Counter(labels)
        
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        augmented_domains = list(domains)
        
        for label, count in label_counts.items():
            if count < target_samples_per_class:
                # 需要增强的样本数
                needed_samples = target_samples_per_class - count
                
                # 获取该类别的所有样本
                class_indices = [i for i, l in enumerate(labels) if l == label]
                
                # 为该类别生成增强样本
                for _ in range(needed_samples):
                    # 随机选择一个该类别的样本进行增强
                    idx = random.choice(class_indices)
                    original_text = texts[idx]
                    domain = domains[idx]
                    
                    # 生成增强样本
                    aug_samples = self.domain_adaptive_augment(original_text, domain, augment_rate=0.8)
                    
                    if aug_samples:
                        chosen_aug = random.choice(aug_samples)
                        augmented_texts.append(chosen_aug)
                        augmented_labels.append(label)
                        augmented_domains.append(domain)
        
        return augmented_texts, augmented_labels, augmented_domains


# 数据增强的便捷函数

def create_augmenter(domain_name: str = None) -> TextAugmentation:
    """
    创建针对特定域的数据增强器
    
    Args:
        domain_name: 域名，如 'HuffPost', 'Amazon' 等
        
    Returns:
        配置好的数据增强器
    """
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
    
    # 根据域名调整配置
    if domain_name:
        domain_type = TextAugmentation().get_domain_type(domain_name)
        if domain_type == 'intent':
            # 意图检测数据集通常文本较短，调整参数
            config['synonym_replacement']['max_replacements'] = 2
            config['random_insertion']['max_insertions'] = 1
            config['random_deletion']['prob'] = 0.05
        elif domain_type == 'news':
            # 新闻数据集文本较长，可以更激进的增强
            config['synonym_replacement']['max_replacements'] = 4
            config['back_translation']['prob'] = 0.25
    
    return TextAugmentation(config)


def augment_episode_data(support_set: List[Dict], query_set: List[Dict], 
                        augmenter: TextAugmentation, 
                        episode: int = 0, max_episodes: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    为episode数据进行增强
    
    Args:
        support_set: 支持集
        query_set: 查询集
        augmenter: 数据增强器
        episode: 当前episode
        max_episodes: 总episode数
        
    Returns:
        增强后的(support_set, query_set)
    """
    # 只对支持集进行增强，保持查询集不变用于公平评估
    aug_support_set = []
    
    for sample in support_set:
        text = sample["text"]
        label = sample["label"]
        domain = sample.get("domain", "unknown")
        
        # 保留原始样本
        aug_support_set.append(sample)
        
        # 根据episode进度决定增强强度
        progress = episode / max_episodes if max_episodes > 0 else 0.5
        augment_rate = 0.1 + 0.4 * progress  # 从10%到50%
        
        if random.random() < augment_rate:
            # 生成增强样本
            aug_texts = augmenter.domain_adaptive_augment(text, domain, augment_rate=0.6)
            
            # 限制增强数量，避免支持集过大
            max_aug = max(1, int(2 * progress))
            aug_texts = aug_texts[:max_aug]
            
            for aug_text in aug_texts:
                aug_sample = sample.copy()
                aug_sample["text"] = aug_text
                aug_support_set.append(aug_sample)
    
    return aug_support_set, query_set