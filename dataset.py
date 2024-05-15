import pandas as pd
from transformers import InputExample, PreTrainedTokenizer
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Tokenize text_a and text_b columns in parallel
        tokenized = self.tokenizer(
            self.data['text_a'].astype(str).tolist(),
            self.data['text_b'].astype(str).tolist(),
            truncation='only_second',
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Iterate over the tokenized batches
        for i in range(len(self.data)):
            input_ids_a = tokenized['input_ids'][i]
            input_ids_b = tokenized['input_ids'][i]
            attention_mask_a = tokenized['attention_mask'][i]
            attention_mask_b = tokenized['attention_mask'][i]

            example = InputExample(
                guid=i,
                text_a=str(self.data['text_a'][i]),
                text_b=str(self.data['text_b'][i]),
                label=None
            )

            self.examples.append((example, input_ids_a, input_ids_b, attention_mask_a, attention_mask_b))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example, input_ids_a, input_ids_b, attention_mask_a, attention_mask_b = self.examples[idx]
        return {
            "example": example,
            "input_ids_a": input_ids_a,
            "input_ids_b": input_ids_b,
            "attention_mask_a": attention_mask_a,
            "attention_mask_b": attention_mask_b
        }
    