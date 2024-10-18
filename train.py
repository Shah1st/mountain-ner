import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn.functional as F
from datasets import load_dataset, Dataset, load_from_disk
from transformers import DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

train_path = 'data/final_train_dataset'
val_path = 'data/final_val_dataset'
test_path = 'data/final_test_dataset'

def main():

    final_train_dataset.load_from_disk(train_path)
    final_val_dataset.load_from_disk(val_path)
    final_test_dataset.load_from_disk(test_path)

    def convert_to_bio_format(example):
        new_tags = []
        prev_tag = 0
        
        for idx, tag in enumerate(example['fine_ner_tags']):
            if tag == 24:
                if prev_tag == 24:
                    new_tags.append("I-MOUNTAIN")  # Inside mountain entity
                else:
                    new_tags.append("B-MOUNTAIN")  # Beginning of mountain entity
            else:
                new_tags.append("O")  # Outside any entity
            prev_tag = tag
        
        example['bio_tags'] = new_tags
        return example
    
    
    final_train_dataset = final_train_dataset.map(convert_to_bio_format)
    final_val_dataset = final_val_dataset.map(convert_to_bio_format)
    final_test_dataset = final_test_dataset.map(convert_to_bio_format)
    
    # A mapping for BIO tags to integer labels
    tag2id = {
        "O": 0,  # Non-entity tokens
        "B-MOUNTAIN": 1,  # Beginning of a mountain entity
        "I-MOUNTAIN": 2,  # Inside a mountain entity
    }
    
    # Convert bio_tags from string format to integer format
    def convert_tags_to_ids(example):
        example["bio_tags"] = [tag2id[tag] for tag in example["bio_tags"]]
        return example
    
    final_train_dataset = final_train_dataset.map(convert_tags_to_ids)
    final_val_dataset = final_val_dataset.map(convert_tags_to_ids)
    final_test_dataset = final_test_dataset.map(convert_tags_to_ids)
    
    
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    
    def custom_tokenize_and_align_labels(dataset, tokenizer, max_length=128):
        """
        Function to tokenize sentences and align NER labels with subword tokens.
        Args:
            dataset: The dataset containing tokens and labels to tokenize and align.
            tokenizer: The pre-trained BERT tokenizer.
            max_length: Maximum length of tokenized input.
        Returns:
            tokenized_dataset: The dataset with tokenized inputs and aligned labels.
        """
        def tokenize_and_align(example):
            # Tokenize the input tokens (words) while preserving word boundaries
            tokenized_inputs = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",  # Pad sequences to max_length
                max_length=max_length  # Ensure max length
            )
            
            # Align labels with subword tokens
            word_ids = tokenized_inputs.word_ids()  # Map each token to its original word
            aligned_labels = []
            prev_word_id = None  # To track the previous word ID
            
            for idx, word_id in enumerate(word_ids):
                if word_id is None:  # Special tokens like [CLS], [SEP]
                    aligned_labels.append(-100)  # Ignore labels for special tokens
                elif word_id != prev_word_id:
                    # Assign the original label for the first subword token
                    aligned_labels.append(example["bio_tags"][word_id])
                    prev_word_id = word_id
                else:
                    # For subword tokens, we ignore the label by assigning -100
                    aligned_labels.append(-100)
            
            tokenized_inputs["labels"] = aligned_labels  # Add aligned labels to the tokenized input
            return tokenized_inputs
        
        tokenized_dataset = dataset.map(tokenize_and_align, batched=False)
        
        return tokenized_dataset
    
    
    tokenized_train_dataset = custom_tokenize_and_align_labels(final_train_dataset, tokenizer)
    tokenized_val_dataset = custom_tokenize_and_align_labels(final_val_dataset, tokenizer)
    tokenized_test_dataset = custom_tokenize_and_align_labels(final_test_dataset, tokenizer)
    
    # The format to PyTorch to prepare for training
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    
    
    def updated_compute_metrics(predictions_and_labels):
        """
        Compute metrics across all classes (O, B-MOUNTAIN, I-MOUNTAIN).
        Returns precision, recall, F1 score for all classes, as well as macro-average and accuracy.
        """
        # Unpack predictions and true labels
        predictions, labels = predictions_and_labels
        
        # Convert logits to predicted class labels
        predicted_labels = np.argmax(predictions, axis=2)
        
        # Flatten the predictions and labels for computing metrics
        true_labels = []
        pred_labels = []
        
        for pred_seq, label_seq in zip(predicted_labels, labels):
            for pred_label, true_label in zip(pred_seq, label_seq):
                if true_label != -100:  # Ignore padding and special tokens
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
        
        # Metrics for all classes
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0, 1, 2])
        
        # Macro-average F1 score (average of F1 across classes)
        macro_f1 = np.mean(f1)
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        
        return {
            "precision_'O'": precision[0],  # For 'O'
            "recall_'O'": recall[0],        # For 'O'
            "f1_'O'": f1[0],                # For 'O'
            "precision_'B-MOUNTAIN'": precision[1],  # For 'B-MOUNTAIN'
            "recall_'B-MOUNTAIN'": recall[1],        # For 'B-MOUNTAIN'
            "f1_'B-MOUNTAIN'": f1[1],                # For 'B-MOUNTAIN'
            "precision_'I-MOUNTAIN'": precision[2],  # For 'I-MOUNTAIN'
            "recall_'I-MOUNTAIN'": recall[2],        # For 'I-MOUNTAIN'
            "f1_'I-MOUNTAIN'": f1[2],                # For 'I-MOUNTAIN'
            "macro_f1": macro_f1,               # Macro-average F1 score across all classes
            "accuracy": accuracy                # Overall accuracy
        }
    
    
    
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER", num_labels=3, ignore_mismatched_sizes=True)
    
    
    
    class FocalLossTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.gamma = kwargs.pop('gamma', 2.0)  # Focal Loss gamma parameter
            self.alpha = kwargs.pop('alpha', None)  # alpha parameter for Focal Loss
            
            super().__init__(*args, **kwargs)
    
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")  # Get true labels
            # Forward pass of the model
            outputs = model(**inputs)
            logits = outputs.get('logits')
            
            # Compute Focal Loss
            loss = self.focal_loss(logits, labels)
    
            return (loss, outputs) if return_outputs else loss
    
        def focal_loss(self, logits, labels):
            ce_loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), 
                                      labels.view(-1), 
                                      reduction='none')
            pt = torch.exp(-ce_loss)
            
            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                at = alpha.gather(0, labels.view(-1))
                focal_loss = at * (1 - pt) ** self.gamma * ce_loss
            else:
                focal_loss = (1 - pt) ** self.gamma * ce_loss
    
            return focal_loss.mean()
    
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",            # Directory to save the results
        evaluation_strategy="epoch",       # Evaluation at each epoch
        learning_rate=2e-5,                # Learning rate
        per_device_train_batch_size=16,    # Batch size for training
        per_device_eval_batch_size=16,     # Batch size for evaluation
        num_train_epochs=5,                # Number of training epochs
        weight_decay=0.01,                 # L2 regularization coefficient
    )
    
    
    # Initialize Trainer
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,   # Tokenized training dataset
        eval_dataset=tokenized_val_dataset,      # Tokenized validation dataset
        data_collator=data_collator,             # Data collator for NER
        tokenizer=tokenizer,                     # BERT tokenizer
        compute_metrics=updated_compute_metrics   # Function to compute metrics
    )
    
    trainer.train()
    
    evaluation_results = trainer.evaluate(tokenized_test_dataset)
    
    print(evaluation_results)
    
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')

if __name__ == "__main__":
    main()

