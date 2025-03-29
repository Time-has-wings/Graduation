from paddlenlp.transformers.llama.modeling import LlamaDecoderLayer, LlamaRMSNorm, LlamaLMHead, LlamaConfig, LlamaPretrainingCriterion
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn as nn

class SpecificModelGPU0(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer0 = LlamaDecoderLayer(config)
        self.layer1 = LlamaDecoderLayer(config)
    
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        x = self.layer0(x)
        x = self.layer1(x)
        return x

class SpecificModelGPU1(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer2 = LlamaDecoderLayer(config)
        self.layer3 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class SpecificModelGPU2(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer4 = LlamaDecoderLayer(config)
        self.layer5 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class SpecificModelGPU3(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer6 = LlamaDecoderLayer(config)
        self.layer7 = LlamaDecoderLayer(config)
        self.norm = self.lm_head = None
        if (config.num_hidden_layers == 8):
            self.norm = LlamaRMSNorm(config)
            self.lm_head = LlamaLMHead(config)
    
    def forward(self, x):
        x = self.layer6(x)
        x = self.layer7(x)
        if (self.norm):
            x = self.norm(x)
            x = self.lm_head(x)
        return x

class SpecificModelGPU4(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer8 = LlamaDecoderLayer(config)
        self.layer9 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer8(x)
        x = self.layer9(x)
        return x
    
class SpecificModelGPU5(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer10 = LlamaDecoderLayer(config)
        self.layer11 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer10(x)
        x = self.layer11(x)
        return x
    
class SpecificModelGPU6(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer12 = LlamaDecoderLayer(config)
        self.layer13 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer12(x)
        x = self.layer13(x)
        return x
    
class SpecificModelGPU7(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer14 = LlamaDecoderLayer(config)
        self.layer15 = LlamaDecoderLayer(config)
        self.norm = LlamaRMSNorm(config)
        self.lm_head = LlamaLMHead(config)
    
    def forward(self, x):
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

class SpecificModel(PipelineLayer):
    def __init__(self, config:LlamaConfig, num_stages, topology, **kwargs):
        if (config.num_hidden_layers == 8):
            descs = [
                LayerDesc(SpecificModelGPU0, config),
                LayerDesc(SpecificModelGPU1, config),
                LayerDesc(SpecificModelGPU2, config),
                LayerDesc(SpecificModelGPU3, config),
            ]
        elif (config.num_hidden_layers == 16):
            descs = [
                LayerDesc(SpecificModelGPU0, config),
                LayerDesc(SpecificModelGPU1, config),
                LayerDesc(SpecificModelGPU2, config),
                LayerDesc(SpecificModelGPU3, config),
                LayerDesc(SpecificModelGPU4, config),
                LayerDesc(SpecificModelGPU5, config),
                LayerDesc(SpecificModelGPU6, config),
                LayerDesc(SpecificModelGPU7, config),
            ]
        else:
            raise ValueError("[guangming] Unsupported number of hidden layers.")
        super().__init__(
            layers=descs,
            num_stages=num_stages,
            topology=topology,
            loss_fn=LlamaPretrainingCriterion(config),
            **kwargs
        )
        
class PipelineModel(PipelineLayer):
    def __init__(self, config:LlamaConfig, num_stages, topology, **kwargs):
        descs = [
            LayerDesc(nn.Embedding, config.vocab_size, config.hidden_size),
            *[LayerDesc(LlamaDecoderLayer, config) for _ in range(config.num_hidden_layers)],
            LayerDesc(LlamaRMSNorm, config),
            LayerDesc(LlamaLMHead, config),
        ]
        super().__init__(
            layers=descs,
            num_stages=num_stages,
            topology=topology,
            loss_fn=LlamaPretrainingCriterion(config),
            **kwargs
        )
        
class pp4layer16GPU0(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer0 = LlamaDecoderLayer(config)
        self.layer1 = LlamaDecoderLayer(config)
        self.layer2 = LlamaDecoderLayer(config)
        self.layer3 = LlamaDecoderLayer(config)
    
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class pp4layer16GPU1(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer4 = LlamaDecoderLayer(config)
        self.layer5 = LlamaDecoderLayer(config)
        self.layer6 = LlamaDecoderLayer(config)
        self.layer7 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
class pp4layer16GPU2(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer8 = LlamaDecoderLayer(config)
        self.layer9 = LlamaDecoderLayer(config)
        self.layer10 = LlamaDecoderLayer(config)
        self.layer11 = LlamaDecoderLayer(config)
    
    def forward(self, x):
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x

class pp4layer16GPU3(nn.Layer):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.layer12 = LlamaDecoderLayer(config)
        self.layer13 = LlamaDecoderLayer(config)
        self.layer14 = LlamaDecoderLayer(config)
        self.layer15 = LlamaDecoderLayer(config)
        self.norm = LlamaRMSNorm(config)
        self.lm_head = LlamaLMHead(config)
    
    def forward(self, x):
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x
    
class pp4layer16Model(PipelineLayer):
    def __init__(self, config:LlamaConfig, num_stages, topology, **kwargs):
        descs = [
            LayerDesc(pp4layer16GPU0, config),
            LayerDesc(pp4layer16GPU1, config),
            LayerDesc(pp4layer16GPU2, config),
            LayerDesc(pp4layer16GPU3, config),
        ]
        super().__init__(
            layers=descs,
            num_stages=num_stages,
            topology=topology,
            loss_fn=LlamaPretrainingCriterion(config),
            **kwargs
        )