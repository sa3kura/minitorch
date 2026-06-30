#!/usr/bin/env python
from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import pickle
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps


def get_dataset(dataset_name, model_max_length):
    """
    Load and preprocess IWSLT (de-en) dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load
        model_max_length (int): Maximum sequence length for filtering examples

    Returns:
        tuple: (dataset, src_key, tgt_key) where:
            - dataset: Dictionary with 'train', 'validation', 'test' splits
            - src_key (str): Source language key ('de')
            - tgt_key (str): Target language key ('en')
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]  # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Train and save a ByteLevelBPETokenizer on the provided dataset.
    
    Args:
        examples (list): Dataset examples for tokenizer training
        vocab_size (int): Desired vocabulary size
        src_key (str): Source language key in examples
        tgt_key (str): Target language key in examples
        workdir (str): Directory to save tokenizer files

    Returns:
        AutoTokenizer: Trained tokenizer with special tokens
                      (e.g., "<eos_de>", "<eos_en>", "<pad>")
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepare a batch of examples for model training or evaluation.
    
    Args:
        examples (list): List of examples to process
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding texts
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors

    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequences of shape (batch_size, model_max_length-1)
            - labels: Target sequences of shape (batch_size, model_max_length-1)
            - label_token_weights: Weight mask for loss computation of shape (batch_size, model_max_length-1)
            
    Note:
        input_ids format: <de_tokens> + <de_eos> + <en_tokens> + <en_eos> + <pad>
        labels: Next tokens to predict (shifted by 1)
        label_token_weights: 0 for source tokens, 1 for target tokens
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    Compute MLE loss for a batch of examples.
    
    Args:
        batch (dict): Batch data containing 'input_ids', 'labels', 'label_token_weights'
        model (DecoderLM): Language model for prediction

    Returns:
        Tensor: Average loss across all target tokens
    """

    idx = batch['input_ids']
    idx.requires_grad_(True)
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    targets.requires_grad_(True)
    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    return ((loss * label_token_weights).sum() / label_token_weights.sum())


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower()
        if value in ('1', 'true', 't', 'yes', 'y'):
            return True
        if value in ('0', 'false', 'f', 'no', 'n'):
            return False
    return bool(value)


def tensor_to_numpy(value):
    return np.array(value.to_numpy(), copy=True)


def get_model_parameters(model):
    return dict(model.named_parameters())


def get_optimizer_state(optimizer, model_params):
    param_id_to_name = {id(param): name for name, param in model_params.items()}
    states = {}
    for param_id, state in optimizer._states.items():
        param_name = param_id_to_name.get(param_id)
        if param_name is None:
            continue

        saved_state = {}
        for key, value in state.items():
            if hasattr(value, 'to_numpy'):
                saved_state[key] = tensor_to_numpy(value)
            else:
                saved_state[key] = value
        states[param_name] = saved_state

    return {
        'lr': optimizer.lr,
        'beta1': optimizer.beta1,
        'beta2': optimizer.beta2,
        'eps': optimizer.eps,
        'states': states
    }


def save_checkpoint(model, optimizer, epoch, metrics, best_bleu, workdir,
                    is_best=False):
    model_params = get_model_parameters(model)
    model_state = {
        name: tensor_to_numpy(param.value)
        for name, param in model_params.items()
        if param.value is not None and hasattr(param.value, 'to_numpy')
    }

    checkpoint = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': get_optimizer_state(optimizer, model_params),
        'metrics': metrics,
        'best_bleu': best_bleu,
        'timestamp': time.time()
    }

    checkpoint_path = os.path.join(workdir, f'checkpoint_epoch_{epoch}.pkl')
    latest_path = os.path.join(workdir, 'checkpoint_latest.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    if is_best:
        best_path = os.path.join(workdir, 'checkpoint_best.pkl')
        with open(best_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    model_params = get_model_parameters(model)
    for name, value in checkpoint.get('model_state', {}).items():
        if name not in model_params:
            continue
        param = model_params[name]
        restored = minitorch.tensor_from_numpy(
            np.array(value, copy=True),
            backend=param.value.backend,
            requires_grad=True)
        param.update(restored)

    optimizer_state = checkpoint.get('optimizer_state', {})
    optimizer.lr = optimizer_state.get('lr', optimizer.lr)
    optimizer.beta1 = optimizer_state.get('beta1', optimizer.beta1)
    optimizer.beta2 = optimizer_state.get('beta2', optimizer.beta2)
    optimizer.eps = optimizer_state.get('eps', optimizer.eps)

    name_to_param = get_model_parameters(model)
    restored_states = {id(param): {} for param in optimizer.parameters}
    for name, state in optimizer_state.get('states', {}).items():
        param = name_to_param.get(name)
        if param is None:
            continue

        restored_state = {}
        for key, value in state.items():
            if key in ('exp_avg', 'exp_avg_sq'):
                restored_state[key] = minitorch.tensor_from_numpy(
                    np.array(value, copy=True),
                    backend=param.value.backend,
                    requires_grad=False)
            else:
                restored_state[key] = value
        restored_states[id(param)] = restored_state
    optimizer._states = restored_states

    return checkpoint


def cleanup_checkpoints(workdir, keep_last_n):
    if keep_last_n is None or keep_last_n <= 0:
        return

    checkpoint_files = []
    for filename in os.listdir(workdir):
        if filename.startswith('checkpoint_epoch_') and filename.endswith('.pkl'):
            epoch = filename[len('checkpoint_epoch_'):-len('.pkl')]
            if epoch.isdigit():
                checkpoint_files.append((int(epoch), filename))

    checkpoint_files.sort()
    for _, filename in checkpoint_files[:-keep_last_n]:
        os.remove(os.path.join(workdir, filename))


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    Train the model on provided examples.
    
    Args:
        model (DecoderLM): Model to train
        optimizer (Adam): Optimizer for parameter updates
        examples (list): Training dataset examples
        n_samples (int): Number of random samples to use
        collate_fn (callable): Function to collate examples into batches
        batch_size (int): Number of examples per batch
        desc (str): Description for progress bar
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]
    losses = []
    total_tokens = 0
    total_compute_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_optimizer_time = 0.0
    train_start = time.time()

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        batch_time = time.time() - t0
        batch_tokens = np.prod(batch['input_ids'].shape)
        losses.append(loss.item())
        total_tokens += batch_tokens
        total_compute_time += batch_time
        total_forward_time += t1 - t0
        total_backward_time += t2 - t1
        total_optimizer_time += t3 - t2
        prog_bar.set_postfix(
            tokens_per_sec=batch_tokens / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)

    train_time = time.time() - train_start
    return {
        'train_loss': float(np.mean(losses)) if losses else None,
        'train_time_sec': train_time,
        'train_compute_time_sec': total_compute_time,
        'forward_time_sec': total_forward_time,
        'backward_time_sec': total_backward_time,
        'optimizer_time_sec': total_optimizer_time,
        'train_tokens_per_sec': (
            float(total_tokens / total_compute_time) if total_compute_time > 0 else None
        ),
        'n_train_batches': len(losses),
        'n_train_examples': len(examples)
    }


def generate(
    model,
    examples,
    src_key,
    tgt_key,
    tokenizer,
    model_max_length,
    backend,
    desc
):
    """
    Generate target sequences for source sequences using argmax decoding.
    
    Args:
        model (DecoderLM): Model for generation
        examples (list): Dataset examples containing source sequences
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding/decoding
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors
        desc (str): Description for progress bar

    Returns:
        list: Generated target sequences
    """

    model.eval()
    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        # Run generation for every single example

        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            # BEGIN ASSIGN3_4
            # run the model with current token_ids, and predict the next token (gen_id)
            # hint: obtain the logits of next token, and take the argmax.
            gen_id = 0
            
            input_tensor = minitorch.tensor([token_ids], backend=backend)
            logits = model(input_tensor)
            
            logits_np = logits.to_numpy()
            last_token_logits_np = logits_np[0, -1, :]
            
            gen_id = int(np.argmax(last_token_logits_np))
            # END ASSIGN3_4

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluate BLEU score for generated sentences against target sentences.
    
    Args:
        examples (list): Dataset examples containing target sentences
        gen_sents (list): Generated sentences to evaluate
        tgt_key (str): Key for target texts in examples

    Returns:
        dict: Dictionary containing BLEU score
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }


def main(
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=40,
    n_epochs=20,
    batch_size=128,
    learning_rate=0.002,
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    seed=11111,
    use_fused_kernel=False,
    resume=False,
    checkpoint_path=None,
    keep_last_n_checkpoints=3,
    bleu_eval_split='test'
):
    """
    Train and evaluate a decoder-only transformer language model.
    
    Args:
        dataset_name (str): Name of the dataset to use, default 'bbaaaa/iwslt14-de-en-preprocess'
        model_max_length (int): Maximum sequence length, default 40
        n_epochs (int): Number of training epochs, default 20
        batch_size (int): Number of examples per batch, default 128
        learning_rate (float): Learning rate for optimizer, default 0.02
        samples_per_epoch (int): Training samples per epoch, default 20000
        n_vocab (int): Vocabulary size for tokenizer, default 10000
        n_embd (int): Embedding dimension, default 256
        seed (int): Random seed, default 11111
        use_fused_kernel (bool): Whether to use fused CUDA kernels.
        resume (bool): Whether to resume from a saved checkpoint.
        checkpoint_path (str): Specific checkpoint to load. If None, loads latest.
        keep_last_n_checkpoints (int): Number of epoch checkpoints to keep.
        bleu_eval_split (str): Dataset split used for BLEU-based best checkpoint.
    """
    use_fused_kernel = parse_bool(use_fused_kernel)
    resume = parse_bool(resume)
    if keep_last_n_checkpoints is not None:
        keep_last_n_checkpoints = int(keep_last_n_checkpoints)

    np.random.seed(seed)
    random.seed(seed)

    kernel_tag = 'fused' if use_fused_kernel else 'unfused'
    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}_{kernel_tag}'
    os.makedirs(workdir, exist_ok=True)
    metrics_path = os.path.join(workdir, 'training_metrics.json')

    backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab': n_vocab,  # vocab_size
        'n_embd': n_embd,  # n_embed
        'n_head': 8,  # n_head
        'n_positions': model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': 0.1,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend,
        'use_fused_kernel': use_fused_kernel
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    best_bleu = float('-inf')

    if resume:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(workdir, 'checkpoint_latest.pkl')
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = checkpoint['epoch'] + 1
            best_bleu = checkpoint.get('best_bleu', float('-inf'))
            print(f"Resumed from {checkpoint_path} at epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}; starting from scratch.")

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)
    if bleu_eval_split not in dataset:
        raise ValueError(f'Unknown bleu_eval_split: {bleu_eval_split}')

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    if resume and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.setdefault('epochs', [])
        metrics['epochs'] = [
            epoch for epoch in metrics['epochs']
            if epoch.get('epoch', -1) < start_epoch
        ]
        if metrics.get('best') is not None:
            best_bleu = max(best_bleu, metrics['best'].get('bleu', float('-inf')))
    else:
        metrics = {
            'config': {
                'dataset_name': dataset_name,
                'model_max_length': model_max_length,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'samples_per_epoch': samples_per_epoch,
                'n_vocab': n_vocab,
                'n_embd': n_embd,
                'seed': seed,
                'use_fused_kernel': use_fused_kernel,
                'bleu_eval_split': bleu_eval_split
            },
            'data_size': {
                split: len(dataset[split]) for split in dataset.keys()
            },
            'epochs': [],
            'best': None
        }
    save_json(metrics, metrics_path)

    for epoch_idx in range(start_epoch, n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        epoch_start = time.time()
        train_metrics = train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        bleu_start = time.time()
        bleu_examples = dataset[bleu_eval_split]
        gen_sents = generate(
            model=model,
            examples=bleu_examples,
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(bleu_examples, gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        save_json(gen_examples, os.path.join(workdir, f'gen_epoch{epoch_idx}.json'))

        eval_scores = evaluate_bleu(
            examples=bleu_examples, gen_sents=gen_sents, tgt_key=tgt_key)
        current_bleu = float(eval_scores['bleu'])
        bleu_time = time.time() - bleu_start
        is_best = current_bleu > best_bleu
        if is_best:
            best_bleu = current_bleu

        epoch_metrics = {
            'epoch': epoch_idx,
            **train_metrics,
            'bleu': current_bleu,
            'bleu_time_sec': bleu_time,
            'n_bleu_examples': len(bleu_examples),
            'epoch_time_sec': time.time() - epoch_start,
            'is_best': is_best
        }
        print(f'Epoch {epoch_idx}: {eval_scores}')

        metrics['epochs'].append(epoch_metrics)
        if is_best:
            metrics['best'] = {
                'epoch': epoch_idx,
                'bleu': best_bleu,
                'model_path': os.path.join(workdir, 'checkpoint_best.pkl')
            }

        save_json(epoch_metrics, os.path.join(
            workdir, f'eval_results_epoch{epoch_idx}.json'))
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_idx,
            metrics=epoch_metrics,
            best_bleu=best_bleu,
            workdir=workdir,
            is_best=is_best)
        cleanup_checkpoints(workdir, keep_last_n_checkpoints)
        save_json(metrics, metrics_path)


if __name__ == '__main__':
    fire.Fire(main)
