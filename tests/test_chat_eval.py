import torch

from scripts.chat_eval import run_categorical_eval, run_generative_eval


class _MockChatTokenizer:
    def get_bos_token_id(self):
        return 1

    def render_for_completion(self, conversation):
        # return a simple token sequence representing the conversation
        return [1, 10, 20, 30]

    def encode(self, letter):
        # single token per letter
        mapping = {"A": [65], "B": [66], "C": [67], "D": [68]}
        return mapping.get(letter, [99])

    def decode(self, tokens):
        return "42"


class _MockChatModel(torch.nn.Module):
    def __init__(self, vocab_size=128, device="cpu", return_tuple=False):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.return_tuple = return_tuple

    def get_device(self):
        return self.device

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        logits = torch.zeros((bsz, seq_len, self.vocab_size), dtype=torch.float32, device=self.device)
        # Give 'A' (token 65) high logit
        logits[:, :, 65] = 10.0
        if self.return_tuple:
            return logits, None
        return logits


class _MockChatEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(self, encoded_prompt, num_samples=1, max_tokens=16, temperature=0.0, top_k=50):
        # Generate 1 sample with prompt + completion tokens [42]
        out_tokens = [encoded_prompt + [42]]
        return out_tokens, None


class _MockGenerativeTask:
    def __init__(self):
        self.eval_type = "generative"
        self._items = [
            {"messages": [{"role": "user", "content": "What is the answer?"}], "gold": "42"},
            {"messages": [{"role": "user", "content": "What is 2+2?"}], "gold": "4"},
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def evaluate(self, conversation, completion):
        return completion.strip() == conversation["gold"].strip()


class _MockCategoricalTask:
    def __init__(self):
        self.eval_type = "categorical"
        self._items = [
            {"messages": [{"role": "user", "content": "Choose A or B"}], "letters": ["A", "B"], "gold": "A"},
            {"messages": [{"role": "user", "content": "Choose A or B"}], "letters": ["A", "B"], "gold": "B"},
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def evaluate(self, conversation, predicted_letter):
        return predicted_letter == conversation["gold"]


def test_run_generative_eval():
    task = _MockGenerativeTask()
    tokenizer = _MockChatTokenizer()
    model = _MockChatModel()
    engine = _MockChatEngine(model, tokenizer)

    acc = run_generative_eval(
        task_object=task,
        tokenizer=tokenizer,
        model=model,
        engine=engine,
        num_samples=1,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
        max_problems=2,
    )
    # First problem matches "42" (gold "42"), second problem gold is "4" -> 1/2 = 0.5
    assert acc == 0.5


def test_run_categorical_eval():
    task = _MockCategoricalTask()
    tokenizer = _MockChatTokenizer()
    model = _MockChatModel(return_tuple=True)

    acc = run_categorical_eval(
        task_object=task,
        tokenizer=tokenizer,
        model=model,
        batch_size=2,
        max_problems=2,
    )
    # Model always predicts 'A' (highest logit). Problem 1 gold is 'A' (pass), Problem 2 gold is 'B' (fail) -> 1/2 = 0.5
    assert acc == 0.5
