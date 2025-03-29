import paddle.nn as nn 
from paddlenlp.transformers import LlamaForCausalLM
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

class GPU0(nn.Layer):
    def __init__(self, model:LlamaForCausalLM):
        super().__init__()
        self.embed_tokens = model.llama.embed_tokens
        self.layer0 = model.llama.layers[0]
        self.layer1 = model.llama.layers[1]
        for name, param in self.embed_tokens.named_parameters():
            if param._is_initialized():
                print(f'[guangming] has been initialized')
        
    def forward(self, input_ids):   
        x = self.embed_tokens(input_ids)
        x = self.layer0(x)
        x = self.layer1(x)
        return x

class GPU1(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer2 = model.llama.layers[2]
        self.layer3 = model.llama.layers[3]
        
    def forward(self, x):        
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class GPU2(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer4 = model.llama.layers[4]
        self.layer5 = model.llama.layers[5]
        
    def forward(self, x):
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class GPU3(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer6 = model.llama.layers[6]
        self.layer7 = model.llama.layers[7]
        self.norm = None
        if (len(model.llama.layers) == 8):
            self.norm = model.llama.norm
            self.lm_head = model.lm_head
        
    def forward(self, x):
        x = self.layer6(x)
        x = self.layer7(x)
        if (self.norm):
            x = self.norm(x)
            x = self.lm_head(x)
        return x
    
class GPU4(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer8 = model.llama.layers[8]
        self.layer9 = model.llama.layers[9]
        
    def forward(self, x):
        x = self.layer8(x)
        x = self.layer9(x)
        return x
    
class GPU5(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer10 = model.llama.layers[10]
        self.layer11 = model.llama.layers[11]
        
    def forward(self, x):
        x = self.layer10(x)
        x = self.layer11(x)
        return x
    
class GPU6(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer12 = model.llama.layers[12]
        self.layer13 = model.llama.layers[13]
        
    def forward(self, x):
        x = self.layer12(x)
        x = self.layer13(x)
        return x
    
class GPU7(nn.Layer):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.layer14 = model.llama.layers[14]
        self.layer15 = model.llama.layers[15]
        self.norm = model.llama.norm
        self.lm_head = model.lm_head
        
    def forward(self, x):
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

class LlamaPipeDescSpecific(PipelineLayer):
    def __init__(self, model:LlamaForCausalLM, num_stages, topology, **kwargs):
        if (len(model.llama.layers) == 8):
            descs = [LayerDesc(GPU0, model),
                    LayerDesc(GPU1, model),
                    LayerDesc(GPU2, model),
                    LayerDesc(GPU3, model),]
        elif (len(model.llama.layers) == 16):
            descs = [LayerDesc(GPU0, model),
                    LayerDesc(GPU1, model),
                    LayerDesc(GPU2, model),
                    LayerDesc(GPU3, model),
                    LayerDesc(GPU4, model),
                    LayerDesc(GPU5, model),
                    LayerDesc(GPU6, model),
                    LayerDesc(GPU7, model),]
        else:
            raise ValueError("Unsupported model")

        super().__init__(
            layers=descs,
            num_stages=num_stages,
            topology=topology,
            loss_fn=model.criterion,  # 使用LlamaPretrainingCriterion
            **kwargs
        )
        print(f'[guangming] self.run_funciton: {self.run_function}')
        for model in self.run_function:
            if not isinstance(model, nn.Layer):
                raise "model must be paddle.nn.Layer object"
            for name, param in model.named_parameters():
                if not param._is_initialized():
                    param.initialize()
                # else:
                #     print(f'[guangming] has been initialized')
        print(f'[guangming] self._layer_desc: {self._layers_desc}')
        print(f'[guangming] self._layers: {self.layers}')
        print(f'[guangming] self.run_function address')
        for model in self.run_function:
            print(f'[guangming] {id(model)}')
        print(f'[guangming] self._layers address')
        for model in self.layers:
            print(f'[guangming] {id(model)}')
        print(f'[guangming] self._layers_desc address')
        for model in self._layers_desc:
            print(f'[guangming] {id(model)}')
