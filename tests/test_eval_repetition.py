import tiktoken
import torch

from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.tokenizer import RustBPETokenizer
from scripts.eval_repetition import HORIZON, _decode, _presyn, _weight_digest, repetition


def test_equal_length_sequences_and_stops_have_distinct_repetition_accounting():
    repeated = repetition([1, 2] * 12)
    diverse = repetition(list(range(24)))
    assert repeated["length"] == diverse["length"] == 24
    assert repeated["trigram_opportunities"] == diverse["trigram_opportunities"] == 22
    assert repeated["repeated_trigrams"] == 20 and repeated["repeated_per_slot"] == 20 / 22
    assert diverse["repeated_trigrams"] == 0
    stopped = repetition([3, 3])
    assert stopped["length"] == 2 and stopped["repeated_per_slot"] == 0
    assert stopped["repeated_per_emitted_trigram"] is None
    assert repetition([])["trigram_opportunities"] == 0


def test_batched_plain_decoding_matches_serving_greedy_and_preserves_acute_weights():
    specials = {name: 256 + i for i, name in enumerate((
        "<|bos|>", "<|assistant_end|>", "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>",
    ))}
    enc = tiktoken.Encoding(name="repeat-control", pat_str="(?s).",
                            mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens=specials)
    tokenizer = RustBPETokenizer(enc, "<|bos|>")
    cfg = GPTSynapticConfig(sequence_len=64, vocab_size=262, n_layer=1, n_head=1, n_kv_head=1,
                            n_embd=16, syn_cfg=SynapticConfig(native_presyn=False, enable_hebbian=False),
                            final_norm=True)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(17)
        model = GPTSynaptic(cfg).eval()
        model.init_weights()
        model.lm_head.weight.normal_(0, 0.02)
        # Make the tools and end symbols inaccessible in this plain-text parity case.
        model.lm_head.weight[256:].zero_()
    prompts = [list(range(16)), list(range(16, 32))]
    docs = [{"prefix": ids, "seed": 37 + i} for i, ids in enumerate(prompts)]
    before = _weight_digest(model)
    for enabled in (True, False):
        with _presyn(model, enabled):
            batched = _decode(model, tokenizer, docs, 0.0, 0.0)
            for row, ids in zip(batched, prompts, strict=True):
                generated = [tokens[0] for tokens, _ in Engine(model, tokenizer).generate(
                    ids, num_samples=1, max_tokens=HORIZON, temperature=0.0)]
                assert row["output_ids"] == generated
    assert _weight_digest(model) == before and cfg.syn_cfg.enable_presyn
