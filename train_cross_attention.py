import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from models.cross_attention_model import CrossAttentionExtractor
from aokvqa_dataset import AOKVQADataset
from utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return {
        'image':            [item['image'] for item in batch],
        'question':         [item['question'] for item in batch],
        'rationales':       [item['rationales'] for item in batch],
        'mc_answers_index': [item['mc_answers_index'] for item in batch],
    }

def train():
    cfg = load_config("config.yaml")

    # Prepare training log
    log_path = cfg.get('log_path', 'training.log')
    log_f = open(log_path, 'a')
    log_f.write('epoch,vqa_loss,attn_loss,accuracy\n')

    # DataLoader
    train_ds = AOKVQADataset(cfg['aokvqa_dir'], cfg['coco_dir'], split="train")
    loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, collate_fn=collate_fn)

    # Model
    model = CrossAttentionExtractor(
        vision_model_name=cfg['vision_model'],
        text_model_name=cfg['text_model'],
        num_labels=cfg['num_labels']
    ).to(device)

    # Freeze vision and Q-Former
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.qformer.parameters():
        param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=float(cfg['learning_rate']))
    scaler = GradScaler()

    model.train()
    checkpoint_path = cfg.get('checkpoint_path', 'checkpoint_supervised.pt')
    for epoch in range(cfg['num_epochs']):
        total_loss = 0.0
        total_vqa_loss = 0.0
        total_attn_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images      = batch['image']
            questions   = batch['question']
            rationales  = [" ".join(r) if isinstance(r, (list, tuple)) else r
                           for r in batch['rationales']]
            labels      = torch.tensor(batch['mc_answers_index'], dtype=torch.long).to(device)

            optimizer.zero_grad()
            with autocast():
                # Forward: only question + detached image features
                logits, attns = model(images, questions)
                vqa_loss = F.cross_entropy(logits, labels)

                # Attention supervision for head 0
                attn_last = attns[-1]  # [B, heads, seq_len, q_len]
                attn_per_token = attn_last[:, 0:1, :, :].mean(dim=3)  # [B,1,seq_len]

                # Tokenize question and rationale
                txt_inputs = model.tokenizer(
                    [f"Question: {q}" for q in questions],
                    return_tensors="pt", padding=True, truncation=True, max_length=32
                ).to(device)
                rat_inputs = model.tokenizer(
                    rationales,
                    return_tensors="pt", padding=True, truncation=True, max_length=32
                ).to(device)
                token_ids = txt_inputs.input_ids
                rat_ids   = rat_inputs.input_ids

                # Build target mask
                mask = torch.zeros_like(attn_per_token)
                for i in range(token_ids.size(0)):
                    rat_set = set(rat_ids[i].tolist())
                    for j in range(token_ids.size(1)):
                        if token_ids[i, j].item() in rat_set:
                            mask[i, :, j] = 1.0
                mask = mask / (mask.sum(dim=2, keepdim=True) + 1e-8)

                attn_log = (attn_per_token + 1e-8).log()
                attn_loss = F.kl_div(attn_log, mask, reduction='batchmean')

                # compute batch accuracy
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()

                loss = vqa_loss + cfg.get('attn_loss_weight', 1.0) * attn_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # accumulate metrics
            total_vqa_loss += vqa_loss.item()
            total_attn_loss += attn_loss.item()
            total_acc += acc
            num_batches += 1

            total_loss += loss.item()

        # compute and print average metrics
        avg_vqa = total_vqa_loss / num_batches
        avg_attn = total_attn_loss / num_batches
        avg_acc = total_acc / num_batches
        print(
            f"Epoch {epoch+1}/{cfg['num_epochs']} - "
            f"VQA Loss: {avg_vqa:.4f}, Attn Loss: {avg_attn:.4f}, Acc: {avg_acc:.4f}"
        )
        # write metrics to log
        log_f.write(f"{epoch+1},{avg_vqa:.4f},{avg_attn:.4f},{avg_acc:.4f}\n")
        log_f.flush()

        # save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            epoch_ckpt = os.path.splitext(checkpoint_path)[0] + f"_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), epoch_ckpt)
            print(f"Saved periodic checkpoint to {epoch_ckpt}")

    # Save
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    log_f.close()

if __name__ == "__main__":
    train()
