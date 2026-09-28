"""Bound memory for the frozen runtime's native VisionZip attention statistic.

The vision encoder consumes only head-mean attention, then sums query rows.
Keep that matrix, not the H-times-larger per-head attention tensor. This keeps
the original bf16 head mean and the original subsequent query reduction.
"""
import os

import torch

AUDIT = {'chunked_calls': 0, 'verified_calls': 0, 'max_vision_sequence_length': 0}


class HeadMeanAttention:
    def __init__(self, head_mean):
        self.head_mean = head_mean

    def mean(self, dim):
        assert dim == 0
        return self.head_mean


@torch.no_grad()
def head_mean_chunked(query, key, chunk=256):
    assert query.ndim == key.ndim == 4 and query.shape[0] == key.shape[0] == 1
    assert query.shape[1] == key.shape[1]
    result = torch.empty((query.shape[2], key.shape[2]), device=query.device, dtype=query.dtype)
    transposed_key = key.transpose(-1, -2)
    for start in range(0, query.shape[2], chunk):
        logits = torch.matmul(query[:, :, start:start + chunk], transposed_key)
        logits = logits / query.shape[-1] ** .5
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        result[start:start + chunk] = probabilities.squeeze(0).mean(dim=0)
    return result


def install():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.flash_attention import flash_attention_forward
    verify = os.environ.get('CASE_VERIFY_VISIONZIP_CHUNK') == '1'

    def bounded(module, query, key, value, attention_mask, **kwargs):
        if not kwargs.get('enable_visionzip'):
            return flash_attention_forward(module, query, key, value, attention_mask, **kwargs)
        sequence = query.shape[2]
        AUDIT['max_vision_sequence_length'] = max(AUDIT['max_vision_sequence_length'], sequence)
        full_bytes = query.shape[0] * query.shape[1] * sequence * key.shape[2] * query.element_size()
        if full_bytes <= 4 * 2**30 and not verify:
            return flash_attention_forward(module, query, key, value, attention_mask, **kwargs)
        mean = head_mean_chunked(query, key)
        AUDIT['chunked_calls'] += 1
        if verify:
            native = flash_attention_forward(module, query, key, value, attention_mask, **kwargs)
            expected = native[1].mean(dim=0)
            # Test on actual vision activations, including the reduction used for
            # dominant selection. Fail closed if chunking changes these values.
            assert torch.equal(mean, expected), float((mean - expected).abs().max())
            assert torch.equal(mean.sum(dim=0), expected.sum(dim=0))
            AUDIT['verified_calls'] += 1
            return native
        extra = dict(kwargs, enable_visionzip=False, enable_kdvz=True)
        output, _, keys = flash_attention_forward(module, query, key, value, attention_mask, **extra)
        return output, HeadMeanAttention(mean), keys

    ALL_ATTENTION_FUNCTIONS.register('flash_attention_2', bounded)
