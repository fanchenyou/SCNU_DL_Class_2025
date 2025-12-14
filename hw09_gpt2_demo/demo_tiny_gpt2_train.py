import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import random
import time
from tqdm import tqdm
import re
import string
import numpy as np

class LocalSQuADEvaluator:
    def __init__(self):
        """Initialize local SQuAD evaluator"""
        pass
    
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def f1_score(self, prediction, ground_truth):
        """Calculate F1 score between prediction and ground truth."""
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        # Calculate common tokens
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1 * 100  # Convert to percentage
    
    def exact_match_score(self, prediction, ground_truth):
        """Calculate exact match score."""
        return 100.0 if self.normalize_answer(prediction) == self.normalize_answer(ground_truth) else 0.0
    
    def evaluate_batch(self, predictions, references):
        """Evaluate a batch of predictions against references."""
        total_f1 = 0
        total_em = 0
        num_samples = len(predictions)
        
        for pred, ref in zip(predictions, references):
            f1 = self.f1_score(pred, ref)
            em = self.exact_match_score(pred, ref)
            
            total_f1 += f1
            total_em += em
        
        return {
            "exact_match": total_em / num_samples,
            "f1": total_f1 / num_samples
        }


class QAEvaluator:
    def __init__(self, model_name="bert-base-uncased"):
        """Initialize QA evaluator"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Use local SQuAD evaluator
        self.squad_metric = LocalSQuADEvaluator()
        
    def evaluate_on_dataset(self, eval_dataset, max_samples=100):
        """Perform QA evaluation on a dataset"""
        self.model.eval()
        all_predictions = []
        all_answers = []
        
        # Create progress bar for evaluation
        eval_progress = tqdm(eval_dataset[:max_samples], 
                           desc="Evaluating", 
                           leave=False)
        
        with torch.no_grad():
            for i, example in enumerate(eval_progress):
                if i >= max_samples:
                    break
                    
                # Update progress bar description
                eval_progress.set_description(f"Evaluating sample {i+1}/{min(len(eval_dataset), max_samples)}")
                
                # Prepare inputs
                inputs = self.tokenizer(
                    example["question"],
                    example["context"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                # Decode answer
                answer_start = torch.argmax(start_logits)
                answer_end = torch.argmax(end_logits) + 1
                answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
                predicted_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Store prediction and ground truth answer
                all_predictions.append(predicted_answer)
                all_answers.append(example["answer"])
        
        # Calculate metrics using local evaluator
        results = self.squad_metric.evaluate_batch(all_predictions, all_answers)
        return results


def create_toy_qa_dataset(num_samples=5):
    """Create a toy QA dataset with random questions and answers"""
    
    # Define different contexts and their QA pairs
    qa_templates = [
        {
            "context": "The solar system has eight planets, with Earth being the third planet from the Sun.",
            "questions": [
                ("How many planets are in the solar system?", "eight", 4),
                ("Which planet is third from the Sun?", "Earth", 53)
            ]
        },
        {
            "context": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
            "questions": [
                ("Who created Python?", "Guido van Rossum", 48),
                ("When was Python first released?", "1991", 92)
            ]
        },
        {
            "context": "Mount Everest, located in the Himalayas, is the highest mountain on Earth with a height of 8,848 meters.",
            "questions": [
                ("What is the highest mountain on Earth?", "Mount Everest", 0),
                ("How tall is Mount Everest in meters?", "8,848", 89)
            ]
        },
        {
            "context": "The Great Wall of China, built over several dynasties, stretches approximately 21,196 kilometers across northern China.",
            "questions": [
                ("What is the approximate length of the Great Wall of China?", "21,196 kilometers", 65),
                ("Where is the Great Wall of China located?", "northern China", 108)
            ]
        },
        {
            "context": "Artificial Intelligence is a field of computer science that focuses on creating machines capable of intelligent behavior.",
            "questions": [
                ("What field of science does AI belong to?", "computer science", 41),
                ("What is the main focus of Artificial Intelligence?", "creating machines capable of intelligent behavior", 64)
            ]
        }
    ]
    
    # Create dataset by randomly selecting QA pairs
    dataset = []
    for i in range(num_samples):
        # Randomly select a template
        template = random.choice(qa_templates)
        context = template["context"]
        
        # Randomly select a question from the template
        question, answer, answer_start = random.choice(template["questions"])
        
        dataset.append({
            "context": context,
            "question": question,
            "answer": answer,
            "answer_start": answer_start
        })
    
    return dataset


def prepare_dataloader(dataset, batch_size=2):
    """Prepare a DataLoader from the dataset (simplified version)"""
    # Create mock tensors for demonstration
    mock_data = []
    for item in dataset:
        # Create mock tensors (in reality, you'd use the tokenizer)
        mock_data.append({
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "start_positions": torch.tensor([random.randint(0, 50)]),
            "end_positions": torch.tensor([random.randint(60, 100)])
        })
    
    # Simple DataLoader
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    return DataLoader(MockDataset(mock_data), batch_size=batch_size, shuffle=True)


class TrainingLoopWithEval:
    def __init__(self, model, train_loader, eval_dataset, tb_writer):
        self.model = model
        self.train_loader = train_loader
        self.eval_dataset = eval_dataset
        self.tb_writer = tb_writer
        
        # Initialize evaluator
        self.qa_evaluator = QAEvaluator('/data/LLM_MODEL/tiny-gpt2')
        
        # Training hyperparameters
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training statistics
        self.epoch_times = []
        self.batch_times = []
        self.train_losses = []
        self.eval_metrics = []
        
        # Progress tracking
        self.total_epochs = 0
        self.total_batches = 0
    
    def train_epoch(self, epoch, total_epochs):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        epoch_start_time = time.time()
        batch_losses = []
        
        # Create progress bar for this epoch
        epoch_progress = tqdm(enumerate(self.train_loader), 
                             total=len(self.train_loader),
                             desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in epoch_progress:
            batch_start_time = time.time()
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"]
            )
            
            # Calculate loss
            loss = outputs.loss
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)
            
            # Update progress bar
            avg_batch_loss = sum(batch_losses) / len(batch_losses)
            epoch_progress.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{avg_batch_loss:.4f}',
                'time': f'{batch_time:.2f}s'
            })
            
            # Log batch loss to TensorBoard
            step = epoch * len(self.train_loader) + batch_idx
            self.tb_writer.add_scalar('train/loss_batch', batch_loss, step)
            
            self.total_batches += 1
        
        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Completed")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Avg Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def evaluate_and_log(self, epoch, total_epochs):
        """Evaluate and log results to TensorBoard"""
        print(f"\nEvaluating model...")
        
        # Perform QA evaluation on the evaluation dataset
        eval_start_time = time.time()
        eval_results = self.qa_evaluator.evaluate_on_dataset(
            self.eval_dataset, 
            max_samples=5
        )
        eval_time = time.time() - eval_start_time
        
        # Log results to TensorBoard
        self.tb_writer.add_scalar('eval/exact_match', eval_results["exact_match"], epoch)
        self.tb_writer.add_scalar('eval/f1', eval_results["f1"], epoch)
        self.tb_writer.add_scalar('eval/time', eval_time, epoch)
        
        # Add custom evaluation text to TensorBoard
        self._add_eval_text_to_tb(eval_results, epoch)
        
        # Store evaluation metrics
        self.eval_metrics.append({
            'epoch': epoch + 1,
            'exact_match': eval_results["exact_match"],
            'f1': eval_results["f1"],
            'time': eval_time
        })
        
        # Print evaluation results
        print(f"\nEvaluation Results - Epoch {epoch+1}/{total_epochs}")
        print(f"  Exact Match: {eval_results['exact_match']:.2f}%")
        print(f"  F1 Score:     {eval_results['f1']:.2f}%")
        print(f"  Evaluation Time: {eval_time:.1f}s")
        
        return eval_results
    
    def _add_eval_text_to_tb(self, eval_results, epoch):
        """Add evaluation text to TensorBoard's Text tab"""
        eval_text = f"""
        Epoch {epoch + 1} QA Evaluation Results:
        =================================
        Exact Match: {eval_results['exact_match']:.2f}
        F1 Score: {eval_results['f1']:.2f}
        
        Evaluation Dataset Samples:
        """
        
        # Add some example QA pairs from the evaluation dataset
        for i, example in enumerate(self.eval_dataset[:3]):
            eval_text += f"""
            Sample {i + 1}:
            Context: {example['context'][:100]}...
            Question: {example['question']}
            Answer: {example['answer']}
            """
        
        self.tb_writer.add_text('eval/summary', eval_text, epoch)
    
    def train(self, num_epochs, eval_every_n_epochs=1):
        """Main training loop"""
        self.total_epochs = num_epochs
        self.best_f1 = 0.0
        self.best_model_path = None
        
        # Print training configuration
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        print("\nConfiguration:")
        print(f"  Total Epochs: {num_epochs}")
        print(f"  Evaluation Frequency: Every {eval_every_n_epochs} epoch(s)")
        print(f"  Training Samples: {len(self.train_loader.dataset)}")
        print(f"  Evaluation Samples: {len(self.eval_dataset)}")
        print(f"  Batch Size: {self.train_loader.batch_size}")
        print(f"  Optimizer: AdamW (lr={self.optimizer.param_groups[0]['lr']:.0e})")
        
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train for one epoch
            avg_loss = self.train_epoch(epoch, num_epochs)
            self.tb_writer.add_scalar('train/loss_epoch', avg_loss, epoch)
            
            # Evaluate at specified intervals
            if (epoch + 1) % eval_every_n_epochs == 0:
                eval_results = self.evaluate_and_log(epoch, num_epochs)
                
                # Save best model
                if eval_results["f1"] > self.best_f1:
                    self.best_f1 = eval_results["f1"]
                    self.best_model_path = f"best_model_epoch_{epoch+1}_f1_{self.best_f1:.2f}.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss,
                        'f1': self.best_f1,
                        'exact_match': eval_results["exact_match"]
                    }, self.best_model_path)
                    print(f"Saved best model to: {self.best_model_path}")
        
        # Training completion
        total_training_time = time.time() - training_start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        print(f"\nTraining finished successfully!")
        print(f"  Total Training Time: {total_training_time:.1f}s")
        print(f"  Best F1 Score: {self.best_f1:.2f}%")
        print(f"  Best Model: {self.best_model_path}")
        
        self.tb_writer.close()
        print(f"\nTensorBoard logs saved to: {self.tb_writer.log_dir}")
        
        return self.best_f1


# Add Counter class for the local SQuAD evaluator
from collections import Counter


def main():
    # 1. Initialize TensorBoard writer
    tb_writer = SummaryWriter(log_dir='./runs/qa_toy_experiment')
    
    # 2. Create toy datasets
    print("Creating toy datasets...")
    
    # Create training dataset with 5 random QA pairs
    train_dataset = create_toy_qa_dataset(num_samples=5)
    print(f"Created training dataset with {len(train_dataset)} QA pairs")
    
    # Create evaluation dataset with 5 different random QA pairs
    eval_dataset = create_toy_qa_dataset(num_samples=5)
    print(f"Created evaluation dataset with {len(eval_dataset)} QA pairs")
    
    # 3. Display sample QA pairs
    print("\nSample Training QA Pairs:")
    for i, item in enumerate(train_dataset[:2]):
        print(f"  {i+1}. Q: {item['question']}")
        print(f"     A: {item['answer']}")
        print(f"     Context: {item['context'][:60]}...\n")
    
    # 4. Prepare DataLoader (simplified for demonstration)
    train_loader = prepare_dataloader(train_dataset, batch_size=2)
    
    # 5. Initialize model
    print("\nInitializing model...")
    model = AutoModelForQuestionAnswering.from_pretrained("/data/LLM_MODEL/tiny-gpt2")
    
    # 6. Start training
    trainer = TrainingLoopWithEval(
        model=model,
        train_loader=train_loader,
        eval_dataset=eval_dataset,
        tb_writer=tb_writer
    )
    
    # Train for 3 epochs, evaluate every epoch
    best_f1 = trainer.train(num_epochs=300, eval_every_n_epochs=1)
    
    print(f"\nBest model achieved F1 score: {best_f1:.2f}%")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    main()