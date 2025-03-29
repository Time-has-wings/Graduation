import paddle.nn as nn 
from paddlenlp.transformers import LlamaForCausalLM
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

class EmbeddingWrapper(nn.Layer):
    def __init__(self, llama_for_causal_lm:LlamaForCausalLM):
        super().__init__()
        self.embed_tokens = llama_for_causal_lm.llama.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)
    
class TransformerWrapper(nn.Layer):
    def __init__(self, llama_for_causal_lm:LlamaForCausalLM, idx):
        super().__init__()
        self.layer = llama_for_causal_lm.llama.layers[idx]

    def forward(self, hidden_states):
        return self.layer(hidden_states)

class RMSNormWrapper(nn.Layer):
    def __init__(self, llama_for_causal_lm:LlamaForCausalLM):
        super().__init__()
        self.norm = llama_for_causal_lm.llama.norm
    
    def forward(self, hidden_states):
        return self.norm(hidden_states)

class LMHeadWrapper(nn.Layer):
    def __init__(self, llama_for_causal_lm:LlamaForCausalLM):
        super().__init__()
        self.lm_head = llama_for_causal_lm.lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)
    
class LlamaPipeDesc(PipelineLayer):
    def __init__(self, llama_for_causal_lm:LlamaForCausalLM, num_stages, topology, **kwargs):
        descs = [LayerDesc(EmbeddingWrapper, llama_for_causal_lm)]
        for idx in range(len(llama_for_causal_lm.llama.layers)):
            descs.append(LayerDesc(TransformerWrapper, llama_for_causal_lm, idx))
        descs.append(LayerDesc(RMSNormWrapper, llama_for_causal_lm))
        descs.append(LayerDesc(LMHeadWrapper, llama_for_causal_lm))

        super().__init__(
            layers=descs,
            num_stages=num_stages,
            topology=topology,
            **kwargs
        )