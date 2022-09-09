import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader


class TBSADataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          padding='max_length',
          return_token_type_ids=False,
          max_length=self.max_len,
          return_attention_mask=True,
          return_tensors='pt',
          truncation='only_first'
        )

        return {
          'review_text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, args):
    ds = TBSADataset(
        texts=df.target_text.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=args.tokenizer,
        max_len=args.max_len
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True
    )


if __name__ == '__main__':
    pass
