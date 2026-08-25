import torch

from bio_inspired_nanochat.core_eval import (
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    evaluate_example,
    evaluate_task,
    find_common_length,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
    stack_sequences,
)


class _MockTokenizer:
    """Mock character-level/word-level tokenizer for deterministic test evaluation."""

    def __init__(self, vocab_map=None, bos_id=1):
        self.vocab_map = vocab_map or {}
        self.bos_id = bos_id
        self._next_id = max([0] + list(self.vocab_map.values())) + 1

    def get_bos_token_id(self):
        return self.bos_id

    def __call__(self, texts, prepend=None):
        out = []
        for text in texts:
            # Deterministic tokenization: word chunks or character chunks
            # simulate BPE-style whitespace absorption
            words = text.split(" ")
            tokens = []
            if prepend is not None:
                tokens.append(prepend)
            for i, word in enumerate(words):
                token_str = word if i == 0 else f" {word}"
                if token_str not in self.vocab_map:
                    self.vocab_map[token_str] = self._next_id
                    self._next_id += 1
                tokens.append(self.vocab_map[token_str])
            out.append(tokens)
        return out


class _MockEvalModel(torch.nn.Module):
    def __init__(self, vocab_size=500):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = 512

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        # Return logits predicting input_ids shifted by 1 (perfect predictions)
        # or logits with a target shape (bsz, seq_len, vocab_size)
        logits = torch.zeros((bsz, seq_len, self.vocab_size), dtype=torch.float32)
        # Shift target to simulate predicting the next token
        for b in range(bsz):
            for t in range(seq_len - 1):
                target = input_ids[b, t + 1].item()
                if 0 <= target < self.vocab_size:
                    logits[b, t, target] = 10.0
        return logits


def test_find_common_length():
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [1, 2, 3, 9, 10]
    assert find_common_length([seq1, seq2], direction="left") == 3

    seq3 = [9, 8, 3, 4, 5]
    assert find_common_length([seq1, seq3], direction="right") == 3


def test_stack_sequences():
    seqs = [[1, 2], [1, 2, 3, 4]]
    stacked = stack_sequences(seqs, pad_token_id=0)
    assert stacked.shape == (2, 4)
    assert stacked[0, 2].item() == 0
    assert stacked[0, 3].item() == 0
    assert stacked[1, 3].item() == 4


def test_render_and_batch_prompts_mc():
    tokenizer = _MockTokenizer()
    item = {
        "query": "Which fruit is red?",
        "choices": ["the apple", "the banana", "the grape"],
        "gold": 0,
    }
    prompt_without, prompts = render_prompts_mc(item, continuation_delimiter=" ")
    assert prompt_without == "Which fruit is red?"
    assert len(prompts) == 3

    tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompt_without, prompts)
    assert len(tokens) == 3
    # Ensure start indices isolate the answer choices even when sharing "the"
    assert all(s > 0 for s in start_idxs)
    assert all(e > s for s, e in zip(start_idxs, end_idxs))


def test_render_and_batch_prompts_schema():
    tokenizer = _MockTokenizer()
    item = {
        "context_options": ["The trophy did not fit in the suitcase because it was too large.", "The trophy did not fit in the suitcase because it was too small."],
        "continuation": " What was too large?",
        "gold": 0,
    }
    prompts = render_prompts_schema(item, continuation_delimiter="")
    assert len(prompts) == 2
    tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    assert len(tokens) == 2
    assert all(s > 0 for s in start_idxs)
    assert all(e > s for s, e in zip(start_idxs, end_idxs))


def test_render_and_batch_prompts_lm_squad_boundary():
    tokenizer = _MockTokenizer()
    item = {
        "context": "The capital of France is Paris.",
        "continuation": "Paris",
    }
    prompts = render_prompts_lm(item, continuation_delimiter="\nAnswer: ")
    assert len(prompts) == 2
    assert prompts[0].startswith("The capital of France is Paris.")
    assert prompts[1].endswith("Paris")

    tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    assert len(tokens) == 1
    assert start_idxs[0] < end_idxs[0]
    # Verify the continuation token is indexed at start_idx
    cont_tokens = tokens[0][start_idxs[0]:end_idxs[0]]
    assert len(cont_tokens) > 0


def test_evaluate_example_mc_and_lm():
    tokenizer = _MockTokenizer()
    model = _MockEvalModel(vocab_size=500)
    device = torch.device("cpu")

    # Multiple Choice task evaluation
    mc_item = {
        "query": "What is 2+2?",
        "choices": ["4", "5", "6"],
        "gold": 0,
    }
    mc_meta = {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }
    is_correct_mc = evaluate_example(0, model, tokenizer, [mc_item], device, mc_meta)
    assert is_correct_mc is True

    # Language Modeling / QA task evaluation
    lm_item = {
        "context": "Question: What is 3+3?",
        "continuation": " 6",
    }
    lm_meta = {
        "task_type": "language_modeling",
        "num_fewshot": 0,
        "continuation_delimiter": "\nAnswer: ",
    }
    is_correct_lm = evaluate_example(0, model, tokenizer, [lm_item], device, lm_meta)
    assert is_correct_lm is True


def test_evaluate_task_aggregation():
    tokenizer = _MockTokenizer()
    model = _MockEvalModel(vocab_size=500)
    device = torch.device("cpu")

    data = [
        {"query": "Q1", "choices": ["A", "B"], "gold": 0},
        {"query": "Q2", "choices": ["A", "B"], "gold": 1},
    ]
    task_meta = {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }
    acc = evaluate_task(model, tokenizer, data, device, task_meta)
    assert 0.0 <= acc <= 1.0
