import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
from typing import List, Dict, Tuple


class MLTCDataset(Dataset):
    """多标签长尾文本分类数据集"""

    def __init__(self,
                 texts: List[str],
                 titles: List[str],
                 labels: List[List[int]],
                 label_freq: Dict[int, int],
                 tokenizer: BertTokenizer,
                 max_length: int = 512,
                 split_threshold: int = 100):

        self.texts = texts
        self.titles = titles
        self.labels = labels
        self.label_freq = label_freq
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_threshold = split_threshold

        # 计算高频/低频标签
        self.hf_labels = [idx for idx, freq in label_freq.items()
                          if freq >= split_threshold]
        self.lf_labels = [idx for idx, freq in label_freq.items()
                          if freq < split_threshold]

        # 创建标签映射
        self.label2idx = {label: i for i, label in enumerate(label_freq.keys())}
        self.idx2label = {i: label for label, i in self.label2idx.items()}

        # 计算HF/LF掩码
        self.hf_mask = self._create_label_mask(self.hf_labels)
        self.lf_mask = self._create_label_mask(self.lf_labels)

    def _create_label_mask(self, target_labels: List[int]) -> torch.Tensor:
        """创建标签掩码"""
        mask = torch.zeros(len(self.label_freq), dtype=torch.bool)
        for label in target_labels:
            if label in self.label2idx:
                mask[self.label2idx[label]] = True
        return mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        title = self.titles[idx] if self.titles[idx] else "[CLS]"
        label_vec = self.labels[idx]

        # 编码文本和标题
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        title_encoding = self.tokenizer(
            title,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )

        # 转换为多标签向量
        label_tensor = torch.zeros(len(self.label_freq), dtype=torch.float)
        for label in label_vec:
            if label in self.label2idx:
                label_tensor[self.label2idx[label]] = 1.0

        # 确定是否为低频样本
        is_low_freq = any(label in self.lf_labels for label in label_vec)

        return {
            'input_ids_text': text_encoding['input_ids'].squeeze(0),
            'attention_mask_text': text_encoding['attention_mask'].squeeze(0),
            'input_ids_title': title_encoding['input_ids'].squeeze(0),
            'attention_mask_title': title_encoding['attention_mask'].squeeze(0),
            'labels': label_tensor,
            'is_low_freq': torch.tensor(is_low_freq, dtype=torch.bool),
            'hf_mask': self.hf_mask,
            'lf_mask': self.lf_mask
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """自定义批处理函数"""
        return {
            'input_ids_text': torch.stack([item['input_ids_text'] for item in batch]),
            'attention_mask_text': torch.stack([item['attention_mask_text'] for item in batch]),
            'input_ids_title': torch.stack([item['input_ids_title'] for item in batch]),
            'attention_mask_title': torch.stack([item['attention_mask_title'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'is_low_freq': torch.stack([item['is_low_freq'] for item in batch]),
            'hf_mask': batch[0]['hf_mask'],  # 所有样本相同
            'lf_mask': batch[0]['lf_mask']  # 所有样本相同
        }