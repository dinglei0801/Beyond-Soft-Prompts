import os
import json
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Tuple
from tqdm import tqdm

def get_data(path, mode='json'):
    result = []
    with open(path, 'r', encoding='utf-8') as src:
        if mode == 'json':
            for line in tqdm(src):
                line = json.loads(line)
                result.append(line)
        else:
            for line in tqdm(src):
                line = line.split('\n')[0]
                result.append(line)
    return result

class CrossDomainDataset(Dataset):
    """跨域数据集，支持多个域的数据"""
    def __init__(self, domain_paths: Dict[str, str]):
        """
        Args:
            domain_paths: 字典，key为域名，value为数据路径
                         例如: {'HuffPost': 'data/HuffPost/few_shot/01/train.json'}
        """
        self.domains = list(domain_paths.keys())
        self.domain_datasets = {}
        self.domain_class_mappings = {}
        
        # 为每个域加载数据
        for domain, path in domain_paths.items():
            df = pd.DataFrame(self._index_subset(path))
            df = df.assign(id=df.index.values)
            df = df.assign(domain=domain)  # 添加域信息
            
            # 为每个域创建独立的类别映射
            unique_characters = sorted(df['class_name'].unique())
            class_name_to_id = {unique_characters[i]: i for i in range(len(unique_characters))}
            df = df.assign(class_id=df['class_name'].apply(lambda c: class_name_to_id[c]))
            
            self.domain_datasets[domain] = df
            self.domain_class_mappings[domain] = class_name_to_id
        
        # 合并所有域的数据
        self.df = pd.concat(list(self.domain_datasets.values()), ignore_index=True)
        self.df = self.df.assign(id=self.df.index.values)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        return {
            'text': row['text'],
            'label': row['class_id'],
            'class_name': row['class_name'],
            'domain': row['domain']
        }

    def __len__(self):
        return len(self.df)
    
    def get_domain_data(self, domain: str):
        """获取特定域的数据"""
        return self.domain_datasets.get(domain, pd.DataFrame())
    
    def get_domain_classes(self, domain: str):
        """获取特定域的类别数量"""
        if domain in self.domain_datasets:
            return len(self.domain_datasets[domain]['class_name'].unique())
        return 0
    
    def get_class_mapping(self, domain: str):
        """获取特定域的类别映射"""
        return self.domain_class_mappings.get(domain, {})

    @staticmethod
    def _index_subset(path):
        """加载单个数据文件"""
        texts = []
        print(f'Indexing {path}...')
        
        datas = get_data(path)
        for line in tqdm(datas):      
            texts.append({
                'text': line["sentence"],
                'class_name': line["label"]
            })
        return texts

class CrossDomainTaskSampler(Sampler):
    """跨域任务采样器"""
    def __init__(self,
                 dataset: CrossDomainDataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 sampling_strategy: str = 'mixed',  # 'mixed', 'single_domain', 'cross_domain'
                 target_domains: List[str] = None):
        """
        Args:
            dataset: CrossDomainDataset实例
            episodes_per_epoch: 每个epoch的episode数量
            n: K-shot的K值
            k: N-way的N值
            q: 查询样本数量
            num_tasks: 每个batch的任务数量
            sampling_strategy: 采样策略
                - 'mixed': 混合采样，支持集和查询集可能来自不同域
                - 'single_domain': 每个任务内的样本都来自同一域
                - 'cross_domain': 支持集和查询集必须来自不同域
            target_domains: 目标域列表，用于测试时的域泛化
        """
        super(CrossDomainTaskSampler, self).__init__(dataset)
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.k = k  # N-way
        self.n = n  # K-shot
        self.q = q  # query数量
        self.num_tasks = num_tasks
        self.sampling_strategy = sampling_strategy
        self.target_domains = target_domains or dataset.domains
        
        self.i_task = 0
        
    def __len__(self):
        return self.episodes_per_epoch
    
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            support_set, query_set = [], []
            episode_labels = []
            episode_domain = None
            
            for task in range(self.num_tasks):
                if self.sampling_strategy == 'single_domain':
                    # 单域采样：所有样本来自同一域
                    domain = random.choice(self.target_domains)
                    episode_domain = domain
                    support_samples, query_samples, labels = self._sample_from_single_domain(domain)
                    
                elif self.sampling_strategy == 'cross_domain':
                    # 跨域采样：支持集和查询集来自不同域
                    support_samples, query_samples, labels = self._sample_cross_domain()
                    
                else:  # mixed
                    # 混合采样：支持集和查询集可能来自不同域
                    support_samples, query_samples, labels = self._sample_mixed()
                
                support_set.extend(support_samples)
                query_set.extend(query_samples)
                if not episode_labels:
                    episode_labels = labels
                    
            yield np.array(support_set), np.array(query_set), episode_labels, episode_domain
    
    #liu shujuji buyongzhege 
    def _sample_from_single_domain(self, domain: str):
        """从单个域采样 - 修改版：增加样本数量检查和回退策略"""
        domain_df = self.dataset.get_domain_data(domain)
        if domain_df.empty:
            raise ValueError(f"No data found for domain: {domain}")
        
        # 随机选择N个类别
        available_classes = domain_df['class_id'].unique()
        if len(available_classes) < self.k:
            raise ValueError(f"Domain {domain} has only {len(available_classes)} classes, need {self.k}")
        
        episode_classes = np.random.choice(available_classes, size=self.k, replace=False)
        
        support_samples = []
        query_samples = []
        episode_labels = []
        
        for class_id in episode_classes:
            class_data = domain_df[domain_df['class_id'] == class_id]
            class_name = class_data.iloc[0]['class_name']
            episode_labels.append(class_name)
            
            # 检查类别数据是否足够 - 新增检查逻辑
            if len(class_data) < self.n + self.q:
                print(f"Warning: Class {class_name} in domain {domain} has only {len(class_data)} samples, "
                      f"need {self.n + self.q}. Using replacement sampling.")
                # 使用有放回采样
                all_samples = class_data.sample(self.n + self.q, replace=True)
                support_data = all_samples.iloc[:self.n]
                query_data = all_samples.iloc[self.n:]
            else:
                # 足够样本，使用无放回采样
                all_samples = class_data.sample(self.n + self.q, replace=False)
                support_data = all_samples.iloc[:self.n]
                query_data = all_samples.iloc[self.n:]
            
            # 采样支持集
            for idx, row in support_data.iterrows():
                support_samples.append({
                    "text": row['text'], 
                    "label": class_name,
                    "domain": domain
                })
            
            # 采样查询集
            for idx, row in query_data.iterrows():
                query_samples.append({
                    "text": row['text'], 
                    "label": class_name,
                    "domain": domain
                })
        
        return support_samples, query_samples, episode_labels
    
    def _sample_from_single_domain_new(self, domain: str):
        """从单个域采样 - 完全按照源码逻辑，但增加容错处理"""
        domain_df = self.dataset.get_domain_data(domain)
        if domain_df.empty:
            raise ValueError(f"No data found for domain: {domain}")
    
        # 随机选择N个类别
        available_classes = domain_df['class_id'].unique()
        if len(available_classes) < self.k:
            raise ValueError(f"Domain {domain} has only {len(available_classes)} classes, need {self.k}")
    
        episode_classes = np.random.choice(available_classes, size=self.k, replace=False)
    
        support_samples = []
        query_samples = []
        episode_labels = []
        support_k = {k: None for k in episode_classes}
    
        # 第一步：先采样所有支持集
        for class_id in episode_classes:
            class_data = domain_df[domain_df['class_id'] == class_id]
            class_name = class_data.iloc[0]['class_name']
            episode_labels.append(class_name)
        
            # 采样支持集
            try:
                support = class_data.sample(self.n, replace=False)
            except ValueError:
                # 如果样本不足，使用有放回采样
                support = class_data.sample(self.n, replace=True)
        
            support_k[class_id] = support
        
            for i, s in support.iterrows():
                support_samples.append({
                    "text": s['text'], 
                    "label": class_name,
                    "domain": domain
                })
    
        # 第二步：采样查询集（关键修复点）
        for class_id in episode_classes:
            class_data = domain_df[domain_df['class_id'] == class_id]
            class_name = class_data.iloc[0]['class_name']
        
            # 计算剩余样本
            remaining_data = class_data[~class_data['id'].isin(support_k[class_id]['id'])]
        
            # === 关键修复：检查剩余样本数量 ===
            if len(remaining_data) >= self.q:
                # 剩余样本足够，无放回采样
                query = remaining_data.sample(self.q, replace=False)
            else:
                # 剩余样本不足，从原始数据有放回采样（源码的隐含行为）
                query = class_data.sample(self.q, replace=True)
        
            for i, q in query.iterrows():
                query_samples.append({
                    "text": q['text'], 
                    "label": class_name,
                    "domain": domain
               })
    
        return support_samples, query_samples, episode_labels
    
    def _sample_cross_domain(self):
        """跨域采样：支持集和查询集来自不同域 - 修改版：增加回退策略"""
        if len(self.target_domains) < 2:
            raise ValueError("Cross-domain sampling requires at least 2 domains")
        
        # 随机选择两个不同的域
        support_domain, query_domain = random.sample(self.target_domains, 2)
        
        # 获取两个域的数据
        support_df = self.dataset.get_domain_data(support_domain)
        query_df = self.dataset.get_domain_data(query_domain)
        
        # 检查是否有足够的类别 - 新增检查逻辑
        support_classes = support_df['class_id'].unique()
        query_classes = query_df['class_id'].unique()
        
        if len(support_classes) < self.k or len(query_classes) < self.k:
            print(f"Warning: Not enough classes for cross-domain sampling. "
                  f"Support domain {support_domain} has {len(support_classes)} classes, "
                  f"Query domain {query_domain} has {len(query_classes)} classes. "
                  f"Falling back to single domain sampling.")
            # 回退到单域采样
            domain = random.choice(self.target_domains)
            return self._sample_from_single_domain(domain)
        
        # 随机选择类别
        support_episode_classes = np.random.choice(support_classes, size=self.k, replace=False)
        query_episode_classes = np.random.choice(query_classes, size=self.k, replace=False)
        
        support_samples = []
        query_samples = []
        episode_labels = []
        
        # 采样支持集
        for i, class_id in enumerate(support_episode_classes):
            class_data = support_df[support_df['class_id'] == class_id]
            class_name = f"class_{i}"  # 使用统一的类别标识
            episode_labels.append(class_name)
            
            # 检查样本数量 - 新增检查逻辑
            if len(class_data) < self.n:
                print(f"Warning: Support class {i} in domain {support_domain} has only {len(class_data)} samples, "
                      f"need {self.n}. Using replacement sampling.")
                support_indices = class_data.sample(self.n, replace=True).index
            else:
                support_indices = class_data.sample(self.n, replace=False).index
                
            for idx in support_indices:
                row = class_data.loc[idx]
                support_samples.append({
                    "text": row['text'], 
                    "label": class_name,
                    "domain": support_domain
                })
        
        # 采样查询集
        for i, class_id in enumerate(query_episode_classes):
            class_data = query_df[query_df['class_id'] == class_id]
            class_name = f"class_{i}"  # 使用统一的类别标识
            
            # 检查样本数量 - 新增检查逻辑
            if len(class_data) < self.q:
                print(f"Warning: Query class {i} in domain {query_domain} has only {len(class_data)} samples, "
                      f"need {self.q}. Using replacement sampling.")
                query_indices = class_data.sample(self.q, replace=True).index
            else:
                query_indices = class_data.sample(self.q, replace=False).index
                
            for idx in query_indices:
                row = class_data.loc[idx]
                query_samples.append({
                    "text": row['text'], 
                    "label": class_name,
                    "domain": query_domain
                })
        
        return support_samples, query_samples, episode_labels
    
    def _sample_mixed(self):
        """混合采样策略"""
        # 随机决定是否使用跨域采样
        if len(self.target_domains) > 1 and random.random() < 0.5:
            try:
                return self._sample_cross_domain()
            except Exception as e:
                print(f"Cross-domain sampling failed: {e}. Falling back to single domain sampling.")
                domain = random.choice(self.target_domains)
                return self._sample_from_single_domain(domain)
        else:
            domain = random.choice(self.target_domains)
            return self._sample_from_single_domain(domain)

def init_cross_domain_dataloader(args, mode, domains=None):
    """初始化跨域数据加载器 - 修改版：默认只使用HuffPost和Amazon"""
    if domains is None:
        # 修改1: 默认只使用样本充足的两个域
        domains = ['BANKING77', 'HWU64', 'OOS', 'Liu']  # 移除了Reuters
        print(f"Using domains: {domains} (Reuters excluded due to insufficient samples)")
    
    # 构建域路径映射
    domain_paths = {}
    base_path = args.dataFile  # 这应该是基础路径，如 'data/HuffPost/few_shot/01'
    
    # 处理base_path，确保它是目录路径 - 修改2: 改进路径处理逻辑
    if base_path.endswith('.json'):
        # 如果传入的是文件路径，提取目录部分
        base_dir = os.path.dirname(base_path)
        # 从文件路径中提取域信息
        path_parts = base_path.split('/')
        current_domain = None
        for part in path_parts:
            if part in ['HuffPost', 'Amazon', 'BANKING77', 'OOS', 'HWU64','Liu']:
                current_domain = part
                break
    else:
        # 如果传入的是目录路径
        base_dir = base_path
        # 从目录路径中提取域信息
        path_parts = base_path.split('/')
        current_domain = None
        for part in path_parts:
            if part in ['HuffPost', 'Amazon', 'BANKING77', 'OOS', 'HWU64','Liu']:
                current_domain = part
                break
    
    for domain in domains:
        if current_domain:
            # 替换域名，保持相同的路径结构
            domain_dir = base_dir.replace(current_domain, domain)
        else:
            # 如果无法提取域信息，构建标准路径
            # 尝试从base_path提取split信息
            if 'few_shot' in base_dir:
                path_parts = base_dir.split('/')
                try:
                    few_shot_idx = path_parts.index('few_shot')
                    if few_shot_idx + 1 < len(path_parts):
                        split_name = path_parts[few_shot_idx + 1]
                    else:
                        split_name = '01'
                except ValueError:
                    split_name = '01'
            else:
                split_name = '01'
            domain_dir = os.path.join('data', domain, 'few_shot', split_name)
        
        # 构建具体的文件路径
        domain_path = os.path.join(domain_dir, f'{mode}.json')
        
        if os.path.exists(domain_path):
            domain_paths[domain] = domain_path
            print(f"Found {domain} data: {domain_path}")
        else:
            print(f"Warning: Path {domain_path} does not exist, skipping domain {domain}")
    
    if not domain_paths:
        raise ValueError(f"No valid domain paths found for mode {mode}. Base path: {base_path}")
    
    # 修改3: 确保至少有一个有效域
    print(f"Successfully loaded {len(domain_paths)} domains: {list(domain_paths.keys())}")
    
    if mode == 'train' or mode == 'valid':
        episode_per_epoch = args.episodeTrain
        sampling_strategy = 'mixed'  # 训练时使用混合策略
    else:
        episode_per_epoch = args.episodeTest
        sampling_strategy = 'single_domain'  # 测试时使用单域策略评估泛化能力
    
    dataset = CrossDomainDataset(domain_paths)
    sampler = CrossDomainTaskSampler(
        dataset, 
        episodes_per_epoch=episode_per_epoch, 
        n=args.numKShot, 
        k=args.numNWay, 
        q=args.numQShot, 
        num_tasks=1,
        sampling_strategy=sampling_strategy,
        target_domains=list(domain_paths.keys())
    )

    return sampler, dataset

# 更新后的标签字典获取函数
def get_cross_domain_label_dict(domain_name):
    """获取跨域标签字典 - 完整版本"""
    label_dicts = {
        # HuffPost 新闻分类
        'HuffPost': {
            'politics': 0, 'wellness': 1, 'entertainment': 2, 'travel': 3, 'style and beauty': 4,
            'parenting': 5, 'healthy living': 6, 'queer voices': 7, 'food': 8, 'business': 9,
            'comedy': 10, 'sports': 11, 'black voices': 12, 'home and living': 13, 'parents': 14,
            'the worldpost': 15, 'weddings': 16, 'women': 17, 'impact': 18, 'divorce': 19, 'crime': 20,
            'media': 21, 'weird': 22, 'green': 23, 'world post': 24, 'religion': 25, 'style': 26,
            'science': 27, 'world': 28, 'taste': 29, 'technology': 30, 'money': 31, 'arts': 32,
            'fifty': 33, 'good': 34, 'arts and culture': 35, 'environment': 36, 'college': 37,
            'latino voices': 38, 'culture and arts': 39, 'education': 40
        },
        
        # Amazon 产品分类
        'Amazon': {
            'Amazon Instant Video': 0, 'Apps for Android': 1, 'Automotive': 2, 'Baby': 3, 'Beauty': 4,
            'Books': 5, 'CDs and Vinyl': 6, 'Cell Phones and Accessories': 7, 'Clothing Shoes and Jewelry': 8,
            'Digital Music': 9, 'Electronics': 10, 'Grocery and Gourmet Food': 11, 'Health and Personal Care': 12,
            'Home and Kitchen': 13, 'Kindle Store': 14, 'Movies and TV': 15, 'Musical Instruments': 16,
            'Office Products': 17, 'Patio Lawn and Garden': 18, 'Pet Supplies': 19, 'Sports and Outdoors': 20,
            'Tools and Home Improvement': 21, 'Toys and Games': 22, 'Video Games': 23
        },
        
        # Reuters 金融新闻分类
        'Reuters': {
            'acquisition': 0, 'aluminium': 1, 'trade deficit': 2, 'cocoa': 3, 'coffee': 4, 'copper': 5,
            'cotton': 6, 'inflation': 7, 'oil': 8, 'profit': 9, 'gdp': 10, 'gold': 11, 'grain': 12,
            'rate': 13, 'industrial production': 14, 'steel': 15, 'unemployment': 16, 'cattle': 17,
            'treasury bank': 18, 'money supply': 19, 'gas': 20, 'orange': 21, 'reserves': 22,
            'retail': 23, 'rubber': 24, 'ship': 25, 'sugar': 26, 'tin': 27, 'tariffs': 28,
            'oils and fats tax': 29, 'producer price wholesale': 30
        },
        
        # 20 Newsgroups 论坛话题分类
        '20News': {
            'alt.atheism': 0, 'comp.graphics': 1, 'comp.os.ms-windows.misc': 2, 'comp.sys.ibm.pc.hardware': 3,
            'comp.sys.mac.hardware': 4, 'comp.windows.x': 5, 'misc.forsale': 6, 'rec.autos': 7,
            'rec.motorcycles': 8, 'rec.sport.baseball': 9, 'rec.sport.hockey': 10, 'sci.crypt': 11,
            'sci.electronics': 12, 'sci.med': 13, 'sci.space': 14, 'soc.religion.christian': 15,
            'talk.politics.guns': 16, 'talk.politics.mideast': 17, 'talk.politics.misc': 18, 'talk.religion.misc': 19
        },
        
        # Banking77 银行业务查询分类
        'BANKING77': {
            'activate_my_card': 0, 'age_limit': 1, 'apple_pay_or_google_pay': 2, 'atm_support': 3,
            'automatic_top_up': 4, 'balance_not_updated_after_bank_transfer': 5, 'balance_not_updated_after_cheque_or_cash_deposit': 6,
            'beneficiary_not_allowed': 7, 'cancel_transfer': 8, 'card_about_to_expire': 9, 'card_acceptance': 10,
            'card_arrival': 11, 'card_delivery_estimate': 12, 'card_linking': 13, 'card_not_working': 14,
            'card_payment_fee_charged': 15, 'card_payment_not_recognised': 16, 'card_payment_wrong_exchange_rate': 17,
            'card_swallowed': 18, 'cash_withdrawal_charge': 19, 'cash_withdrawal_not_recognised': 20,
            'change_pin': 21, 'compromised_card': 22, 'contactless_not_working': 23, 'country_support': 24,
            'declined_card_payment': 25, 'declined_cash_withdrawal': 26, 'declined_transfer': 27,
            'direct_debit_payment_not_recognised': 28, 'disposable_card_limits': 29, 'edit_personal_details': 30,
            'exchange_charge': 31, 'exchange_rate': 32, 'exchange_via_app': 33, 'extra_charge_on_statement': 34,
            'failed_transfer': 35, 'fiat_currency_support': 36, 'get_disposable_virtual_card': 37,
            'get_physical_card': 38, 'getting_spare_card': 39, 'getting_virtual_card': 40,
            'lost_or_stolen_card': 41, 'lost_or_stolen_phone': 42, 'order_physical_card': 43,
            'passcode_forgotten': 44, 'pending_card_payment': 45, 'pending_cash_withdrawal': 46,
            'pending_top_up': 47, 'pending_transfer': 48, 'pin_blocked': 49, 'receiving_money': 50,
            'Refund_not_showing_up': 51, 'request_refund': 52, 'reverted_card_payment?': 53,
            'supported_cards_and_currencies': 54, 'terminate_account': 55, 'top_up_by_bank_transfer_charge': 56,
            'top_up_by_card_charge': 57, 'top_up_by_cash_or_cheque': 58, 'top_up_failed': 59,
            'top_up_limits': 60, 'top_up_reverted': 61, 'topping_up_by_card': 62, 'transaction_charged_twice': 63,
            'transfer_fee_charged': 64, 'transfer_into_account': 65, 'transfer_not_received_by_recipient': 66,
            'transfer_timing': 67, 'unable_to_verify_identity': 68, 'verify_my_identity': 69,
            'verify_source_of_funds': 70, 'verify_top_up': 71, 'virtual_card_not_working': 72,
            'visa_or_mastercard': 73, 'why_verify_identity': 74, 'wrong_amount_of_cash_received': 75,
            'wrong_exchange_rate_for_cash_withdrawal': 76
        },
        
        # HWU64 对话意图分类
        'HWU64': {
            'alarm_query': 0, 'alarm_remove': 1, 'alarm_set': 2, 'audio_volume_down': 3, 'audio_volume_mute': 4,
            'audio_volume_up': 5, 'calendar_query': 6, 'calendar_remove': 7, 'calendar_set': 8,
            'cooking_query': 9, 'cooking_recipe': 10, 'datetime_convert': 11, 'datetime_query': 12,
            'email_addcontact': 13, 'email_query': 14, 'email_querycontact': 15, 'email_sendemail': 16,
            'general_affirm': 17, 'general_commandstop': 18, 'general_confirm': 19, 'general_dontcare': 20,
            'general_explain': 21, 'general_joke': 22, 'general_negate': 23, 'general_praise': 24,
            'general_quirky': 25, 'general_repeat': 26, 'iot_cleaning': 27, 'iot_coffee': 28,
            'iot_hue_lightchange': 29, 'iot_hue_lightdim': 30, 'iot_hue_lighton': 31, 'iot_hue_lightoff': 32,
            'iot_wemo_on': 33, 'iot_wemo_off': 34, 'lists_createoradd': 35, 'lists_query': 36,
            'lists_remove': 37, 'music_likeness': 38, 'music_query': 39, 'music_settings': 40,
            'news_query': 41, 'play_audiobook': 42, 'play_game': 43, 'play_music': 44, 'play_podcasts': 45,
            'play_radio': 46, 'qa_currency': 47, 'qa_definition': 48, 'qa_factoid': 49, 'qa_maths': 50,
            'qa_stock': 51, 'recommendation_events': 52, 'recommendation_locations': 53, 'recommendation_movies': 54,
            'social_post': 55, 'social_query': 56, 'takeaway_order': 57, 'takeaway_query': 58,
            'transport_query': 59, 'transport_taxi': 60, 'transport_ticket': 61, 'transport_traffic': 62,
            'weather_query': 63
        },
        
        # OOS (Clinc150) 意图分类 + 域外检测
        'OOS': {
            'transfer': 0, 'transactions': 1, 'balance': 2, 'freeze_account': 3, 'pay_bill': 4,
            'bill_balance': 5, 'bill_due': 6, 'interest_rate': 7, 'routing': 8, 'min_payment': 9,
            'order_checks': 10, 'pin_change': 11, 'report_fraud': 12, 'account_blocked': 13,
            'spending_history': 14, 'credit_score': 15, 'report_lost_card': 16, 'credit_limit': 17,
            'rewards_balance': 18, 'new_card': 19, 'application_status': 20, 'card_declined': 21,
            'international_fees': 22, 'apr': 23, 'redeem_rewards': 24, 'credit_limit_change': 25,
            'damaged_card': 26, 'replacement_card_duration': 27, 'improve_credit_score': 28, 'expiration_date': 29,
            'recipe': 30, 'restaurant_reviews': 31, 'calories': 32, 'nutrition_info': 33, 'restaurant_suggestion': 34,
            'ingredients_list': 35, 'ingredient_substitution': 36, 'cook_time': 37, 'food_last': 38,
            'meal_suggestion': 39, 'restaurant_reservation': 40, 'confirm_reservation': 41, 'how_busy': 42,
            'cancel_reservation': 43, 'accept_reservations': 44, 'car_rental': 45, 'flight_status': 46,
            'lost_luggage': 47, 'book_flight': 48, 'book_hotel': 49, 'uber': 50, 'schedule_maintenance': 51,
            'last_maintenance': 52, 'jump_start': 53, 'tire_pressure': 54, 'oil_change_when': 55,
            'oil_change_how': 56, 'tire_change': 57, 'pto_request': 58, 'taxes': 59, 'payday': 60,
            'w2': 61, 'pto_balance': 62, 'pto_request_status': 63, 'next_song': 64, 'plug_type': 65,
            'maybe': 66, 'change_language': 67, 'no': 68, 'measurement_conversion': 69, 'timer': 70,
            'make_call': 71, 'text': 72, 'spelling': 73, 'smart_home': 74, 'order': 75, 'shopping_list': 76,
            'shopping_list_update': 77, 'todo_list': 78, 'todo_list_update': 79, 'calendar': 80,
            'calendar_update': 81, 'what_are_your_hobbies': 82, 'order_status': 83, 'reminder': 84,
            'reminder_update': 85, 'repeat': 86, 'yes': 87, 'alarm': 88, 'what_song': 89, 'where_are_you_from': 90,
            'weather': 91, 'date': 92, 'who_made_you': 93, 'pto_used': 94, 'whisper_mode': 95,
            'what_is_your_name': 96, 'time': 97, 'rollover_401k': 98, 'income': 99, 'goodbye': 100,
            'what_can_i_ask_you': 101, 'thank_you': 102, 'sync_device': 103, 'tell_joke': 104,
            'are_you_a_bot': 105, 'meaning_of_life': 106, 'user_name': 107, 'how_old_are_you': 108,
            'volume_up': 109, 'mpg': 110, 'schedule_meeting': 111, 'current_location': 112, 'international_visa': 113,
            'exchange_rate': 114, 'carry_on': 115, 'book_vacation': 116, 'translate': 117, 'define_word': 118,
            'share_location': 119, 'find_phone': 120, 'weather_query': 121, 'traffic': 122, 'directions': 123,
            'gas_type': 124, 'distance': 125, 'insurance': 126, 'insurance_change': 127, 'todo_list_remove': 128,
            'gas': 129, 'vaccines': 130, 'meal_plan': 131, 'what_are_you_made_of': 132, 'direct_deposit': 133,
            'greeting': 134, 'reset_settings': 135, 'volume_down': 136, 'cancel': 137, 'new_year_resolution': 138,
            'fun_fact': 139, 'oos': 140, 'do_you_have_pets': 141, 'why_are_you_called_that': 142,
            'who_do_you_work_for': 143, 'what_are_your_interests': 144, 'calculator': 145, 'definition': 146,
            'next_holiday': 147, 'shopping_list_remove': 148, 'rewards_balance': 149
        },
        
        # Liu57 (SNIPS) 个人助手意图分类
        'liu': {
            'AddToPlaylist': 0, 'BookRestaurant': 1, 'GetWeather': 2, 'PlayMusic': 3, 'RateBook': 4,
            'SearchCreativeWork': 5, 'SearchScreeningEvent': 6
        }
    }
    
    return label_dicts.get(domain_name, {})
