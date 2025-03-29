from paddlenlp.transformers.llama.modeling import LlamaDecoderLayer, LlamaRMSNorm, LlamaLMHead, LlamaConfig, LlamaPretrainingCriterion
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn as nn
from typing import List
from graduation.utils import print

class ModelChunk(nn.Layer):
    def __init__(self, config:LlamaConfig, layer_name_list: List[str]):
        super().__init__()
        self.layers = []
        for layer_name in layer_name_list:
            if layer_name == 'embedding':
                self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            elif layer_name.startswith('decoder'):
                self.layers.append(LlamaDecoderLayer(config))
            elif layer_name == 'norm':
                self.layers.append(LlamaRMSNorm(config))
            elif layer_name == 'lm_head':
                self.layers.append(LlamaLMHead(config))
            else:
                raise ValueError(f"[graduation] Unknown layer name: {layer_name}")
        assert len(self.layers) > 0, f"[graduation] No layers found in {layer_name_list}"
        self.layers = nn.LayerList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def generate_even_pipeline_layer_desc(args):
    num_hidden_layers = args.num_hidden_layers
    pp_degree = args.pp_degree
    assert num_hidden_layers >= pp_degree
    div, mod = num_hidden_layers // pp_degree, num_hidden_layers % pp_degree
    all_layer_num_list = [[] for _ in range(pp_degree)]
    for i in range(pp_degree):
        if i == 0:
            all_layer_num_list[i].append('embedding')
            for j in range(div):
                all_layer_num_list[i].append(f'decoder_{j + i * div}')
        elif i == pp_degree - 1:
            layer_cnt = div if mod == 0 else mod
            for j in range(layer_cnt):
                all_layer_num_list[i].append(f'decoder_{j + i * layer_cnt}')
            all_layer_num_list[i].append('norm')
            all_layer_num_list[i].append('lm_head')
        else:
            for j in range(div):
                all_layer_num_list[i].append(f'decoder_{j + i * div}')
    for i in range(len(all_layer_num_list)):
        print(f"Pipeline stage {i}: {all_layer_num_list[i]}")
    return all_layer_num_list

class PipelineModel(PipelineLayer):
    def __init__(self, all_layer_num_list: List[List[str]], loss_fn, config:LlamaConfig, num_stages, topology, **kwargs):
        desc = []
        for layer_name_list in all_layer_num_list:
            desc.append(LayerDesc(ModelChunk, config, layer_name_list))
        loss_fn = LlamaPretrainingCriterion(config) if loss_fn is None else loss_fn
        
        super().__init__(
            layers=desc,
            num_stages=num_stages,
            topology=topology,
            loss_fn=loss_fn,
            **kwargs
        )