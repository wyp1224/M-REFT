"""Refactored version of main_remastered.py with improved structure and readability"""
from typing import Tuple, Dict, Any
import argparse
import datetime
import logging
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import MKG_Loader
from MKGC_model import M_REFT
from utils import calculate_rank, compute_metrics

# Constants
OMP_NUM_THREADS = 8
SEED = 2024

def setup_environment() -> None:
    """Configure runtime environment settings"""
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(OMP_NUM_THREADS)
    torch.cuda.empty_cache()

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

def configure_logging(exp_name: str, data_name: str, no_write: bool) -> Tuple[logging.Logger, str]:
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not no_write:
        os.makedirs(f"./logs/{exp_name}/{data_name}", exist_ok=True)
        file_handler = logging.FileHandler(f"./logs/{exp_name}/{data_name}/{timestamp}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, timestamp

def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description="M-REFT Model Training")

    # Data configuration
    parser.add_argument('--data', default="DB15K", type=str,
                       choices=["DB15K", "MKG-W", "MKG-Y"],
                       help="Dataset to use for training")

    # Training hyperparameters
    parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate")
    parser.add_argument('--dim', default=256, type=int, help="Embedding dimension")
    parser.add_argument('--num_epoch', default=500, type=int, help="Number of training epochs")
    parser.add_argument('--valid_epoch', default=5, type=int,
                       help="Validate every N epochs")
    parser.add_argument('--batch_size', default=1024, type=int, help="Batch size")

    # Model architecture
    parser.add_argument('--num_layer_enc_ent', default=1, type=int,
                       help="Number of entity encoder layers")
    parser.add_argument('--num_layer_enc_rel', default=1, type=int,
                       help="Number of relation encoder layers")
    parser.add_argument('--num_layer_dec', default=1, type=int,
                       help="Number of decoder layers")
    parser.add_argument('--num_head', default=1, type=int, help="Number of attention heads")
    parser.add_argument('--hidden_dim', default=1024, type=int, help="Hidden dimension size")

    # Regularization
    parser.add_argument('--dropout', default=0.01, type=float, help="General dropout rate")
    parser.add_argument('--emb_dropout', default=0.6, type=float, help="Embedding dropout")
    parser.add_argument('--vis_dropout', default=0.3, type=float, help="Visual features dropout")
    parser.add_argument('--txt_dropout', default=0.1, type=float, help="Text features dropout")
    parser.add_argument('--smoothing', default=0.0, type=float, help="Label smoothing")
    parser.add_argument('--decay', default=0.0, type=float, help="Weight decay")

    # Experiment configuration
    parser.add_argument('--exp', default='MREFT', help="Experiment name")
    parser.add_argument('--no_write', action='store_true', help="Disable logging to files")
    parser.add_argument('--step_size', default=50, type=int, help="Step size for learning rate scheduler")

    return parser.parse_args()

def save_args(args: argparse.Namespace, exp_dir: str) -> None:
    """Save training arguments to file"""
    if not os.path.isfile(f"ckpt/{exp_dir}/args.txt"):
        with open(f"ckpt/{exp_dir}/args.txt", "w") as f:
            for arg_name, arg_value in vars(args).items():
                f.write(f"{arg_name}\t{type(arg_value)}\n")

def initialize_model(args: argparse.Namespace, kg: MKG_Loader) -> M_REFT:
    """Initialize and return the M-REFT model"""
    # Prepare indices
    source_ent_index = torch.arange(kg.num_ent).cuda()
    rel_index = torch.arange(kg.num_rel * 2).cuda()

    # Load neighbor indices
    data_path = f'data/{args.data}'
    with open(f'{data_path}/SREs.pickle', 'rb') as f:
        SREs_index = torch.tensor(np.array(pickle.load(f))).cuda()
    with open(f'{data_path}/neighbors.pkl', 'rb') as f:
        neighbors_index = torch.tensor(pickle.load(f)).cuda()

    return M_REFT(
        num_entities=kg.num_ent,
        num_relations=kg.num_rel,
        embedding_dim=args.dim,
        num_heads=args.num_head,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_layer_enc_ent,
        num_decoder_layers=args.num_layer_dec,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        vis_dropout=args.vis_dropout,
        txt_dropout=args.txt_dropout,
        rel_index=rel_index,
        source_ent_index=source_ent_index,
        SREs_index=SREs_index,
        neighbors_index=neighbors_index,
        score_function=args.data,
    ).cuda()

def evaluate_model(
    model: M_REFT,
    kg: MKG_Loader,
    split: str
) -> Tuple[float, float, float, float, float]:
    """Evaluate model on given data split"""
    model.eval()
    lp_ranks = []
    triplets = getattr(kg, split)

    with torch.no_grad():
        ent_embs, rel_embs = model()

        for h, r, t in tqdm(triplets, desc=f"Evaluating on {split}"):
            # Head prediction
            head_scores = model.score(
                ent_embs, rel_embs,
                torch.tensor([[t, r + kg.num_rel, kg.num_ent]]).cuda()
            )[0].detach().cpu().numpy()
            head_rank = calculate_rank(head_scores, h, kg.filter_dict[(-1, r, t)])

            # Tail prediction
            tail_scores = model.score(
                ent_embs, rel_embs,
                torch.tensor([[h, r, kg.num_ent]]).cuda()
            )[0].detach().cpu().numpy()
            tail_rank = calculate_rank(tail_scores, t, kg.filter_dict[(h, r, -1)])

            lp_ranks.extend([head_rank, tail_rank])

    return compute_metrics(np.array(lp_ranks))

def train_epoch(
    model: M_REFT,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    kg_loader: torch.utils.data.DataLoader
) -> Tuple[float, float, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = total_loss_h = total_loss_t = 0.0

    for b_h, l_h, b_t, l_t in kg_loader:
        ent_embs, rel_embs = model()

        # Tail prediction loss
        scores_t = model.score(ent_embs, rel_embs, b_t.cuda())
        loss_t = loss_fn(scores_t, l_t.cuda())

        # Head prediction loss
        scores_h = model.score(ent_embs, rel_embs, b_h.cuda())
        loss_h = loss_fn(scores_h, l_h.cuda())

        # Backpropagation
        optimizer.zero_grad()
        (loss_h + loss_t).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Update metrics
        total_loss += loss_h.item() + loss_t.item()
        total_loss_h += loss_h.item()
        total_loss_t += loss_t.item()

    return total_loss, total_loss_h, total_loss_t

def save_checkpoint(
    model: M_REFT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    timestamp: str
) -> None:
    """Save model checkpoint"""
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        },
        f"./ckpt/{args.exp}/{args.data}/{timestamp}_{epoch}.ckpt"
    )

def main():
    """Main training procedure"""
    # Initial setup
    setup_environment()
    args = parse_arguments()

    # Prepare directories
    if not args.no_write:
        os.makedirs(f"./result/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./ckpt/{args.exp}/{args.data}", exist_ok=True)
        save_args(args, f"{args.exp}/{args.data}")

    # Configure logging
    logger, timestamp = configure_logging(args.exp, args.data, args.no_write)
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Arguments: {args}")

    # Initialize data and model
    kg = MKG_Loader(args.data, logger)
    kg_loader = torch.utils.data.DataLoader(kg, batch_size=args.batch_size, shuffle=True)
    model = initialize_model(args, kg)

    # Training setup
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.step_size, T_mult=2
    )

    # Training loop
    logger.info("EPOCH\tTOTAL_LOSS\tH_LOSS\tT_LOSS\tTIME")
    best_mrr = 0.0
    start_time = time.time()

    for epoch in range(1, args.num_epoch + 1):
        # Train epoch
        total_loss, loss_h, loss_t = train_epoch(model, optimizer, loss_fn, kg_loader)
        scheduler.step()

        # Log training progress
        elapsed = time.time() - start_time
        logger.info(f"{epoch}\t{total_loss:.6f}\t{loss_h:.6f}\t{loss_t:.6f}\t{elapsed:.6f}s")

        # Validation
        if epoch % args.valid_epoch == 0:
            #for split in ['train', 'valid', 'test']:
            for split in ['valid', 'test']:
                mr, mrr, hit10, hit3, hit1 = evaluate_model(model, kg, split)
                logger.info(f"Link Prediction on {split.capitalize()} Set")
                logger.info(f"MR: {mr}\tMRR: {mrr}\tHit10: {hit10}\tHit3: {hit3}\tHit1: {hit1}")

                # Update best results
                if split == 'valid' and mrr > best_mrr:
                    best_mrr = mrr
                    best_results = (mr, mrr, hit10, hit3, hit1)

            # Save checkpoint periodically
            if epoch % 500 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, args, timestamp)

    # Final results
    logger.info(f"Training completed. Best validation results:")
    logger.info(f"MR: {best_results[0]}\tMRR: {best_results[1]}\t"
                f"Hit10: {best_results[2]}\tHit3: {best_results[3]}\t"
                f"Hit1: {best_results[4]}")

if __name__ == "__main__":
    main()
