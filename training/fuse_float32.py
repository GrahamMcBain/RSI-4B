"""Higher-precision fusion via installed MLX-LM primitives; no training or upload."""
import argparse
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_map,tree_unflatten
from mlx_lm.utils import load,save
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter-path',required=True);p.add_argument('--save-path',required=True);a=p.parse_args()
if Path(a.save_path).exists():raise RuntimeError('Refuse overwrite')
model,tok,config=load(a.model,adapter_path=a.adapter_path,return_config=True,trust_remote_code=False)
model.update(tree_map(lambda x:x.astype(mx.float32) if mx.issubdtype(x.dtype,mx.floating) else x,model.parameters()))
fused=[(n,m.fuse()) for n,m in model.named_modules() if hasattr(m,'fuse')]
if len(fused)!=252:raise ValueError('Unexpected adapted module count')
model.update_modules(tree_unflatten(fused));config['torch_dtype']='float32'
save(Path(a.save_path),a.model,model,tok,config,donate_model=False)
print('Saved float32 standalone fusion; 252 adapted modules',flush=True)
