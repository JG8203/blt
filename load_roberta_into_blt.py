# ~/blt/load_roberta_into_blt.py
import torch
from transformers import AutoModel
from omegaconf import OmegaConf
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.args import TrainArgs
from bytelatent.model.blt import ByteLatentTransformer
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_roberta_to_blt(roberta_model, blt_model):
    roberta_state_dict = roberta_model.state_dict()
    blt_state_dict = blt_model.state_dict()
    
    # This mapping is tricky and depends on the exact layer names in both models.
    # RoBERTa: encoder.layer.N.attention.self.[query/key/value]
    # BLT: global_transformer.layers.N.attention.[wq/wk/wv]
    for i in range(roberta_model.config.num_hidden_layers):
        print(f"Mapping layer {i}...")
        
        # Attention weights
        for part, blt_part in [("query", "wq"), ("key", "wk"), ("value", "wv")]:
            roberta_w_key = f"encoder.layer.{i}.attention.self.{part}.weight"
            roberta_b_key = f"encoder.layer.{i}.attention.self.{part}.bias"
            blt_w_key = f"global_transformer.layers.{i}.attention.{blt_part}.weight"
            blt_b_key = f"global_transformer.layers.{i}.attention.{blt_part}.bias"
            if roberta_w_key in roberta_state_dict:
                blt_state_dict[blt_w_key] = roberta_state_dict[roberta_w_key]
                blt_state_dict[blt_b_key] = roberta_state_dict[roberta_b_key]

        # Attention output projection and norm
        blt_state_dict[f"global_transformer.layers.{i}.attention.wo.weight"] = roberta_state_dict[f"encoder.layer.{i}.attention.output.dense.weight"]
        blt_state_dict[f"global_transformer.layers.{i}.attention.wo.bias"] = roberta_state_dict[f"encoder.layer.{i}.attention.output.dense.bias"]
        blt_state_dict[f"global_transformer.layers.{i}.attention_norm.weight"] = roberta_state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"]
        blt_state_dict[f"global_transformer.layers.{i}.attention_norm.bias"] = roberta_state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"]
        
        # FFN (MLP) layers and norm
        blt_state_dict[f"global_transformer.layers.{i}.feed_forward.w1.weight"] = roberta_state_dict[f"encoder.layer.{i}.intermediate.dense.weight"]
        blt_state_dict[f"global_transformer.layers.{i}.feed_forward.w1.bias"] = roberta_state_dict[f"encoder.layer.{i}.intermediate.dense.bias"]
        blt_state_dict[f"global_transformer.layers.{i}.feed_forward.w2.weight"] = roberta_state_dict[f"encoder.layer.{i}.output.dense.weight"]
        blt_state_dict[f"global_transformer.layers.{i}.feed_forward.w2.bias"] = roberta_state_dict[f"encoder.layer.{i}.output.dense.bias"]
        blt_state_dict[f"global_transformer.layers.{i}.ffn_norm.weight"] = roberta_state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"]
        blt_state_dict[f"global_transformer.layers.{i}.ffn_norm.bias"] = roberta_state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"]

    print("Weight mapping complete.")
    return blt_state_dict

if __name__ == "__main__":
    print("--- Loading pre-trained Tagalog RoBERTa model ---")
    roberta_model = AutoModel.from_pretrained("jcblaise/roberta-tagalog-base", use_safetensors=True)

    print("\n--- Instantiating BLT model structure from config ---")
    cli_args = OmegaConf.create({'config': 'train_blt_from_roberta.yaml'})
    train_args = parse_args_to_pydantic_model(TrainArgs, cli_args=cli_args)
    
    # We need to build the model from the arguments
    blt_model = ByteLatentTransformer(train_args.model)

    print("\n--- Mapping weights from RoBERTa to BLT's Global Transformer ---")
    new_state_dict = map_roberta_to_blt(roberta_model, blt_model)

    print("\n--- Saving initial checkpoint for BLT ---")
    # The training script expects this specific path from the YAML config
    output_path = "models/blt_from_roberta/initial_weights/"
    os.makedirs(output_path, exist_ok=True)
    
    # Save as distributed checkpoint format
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    
    # Load the state dict into the model
    blt_model.load_state_dict(new_state_dict, strict=False)
    
    # Save using distributed checkpoint format
    dcp.save(
        state_dict={"model": blt_model.state_dict()},
        checkpoint_id=output_path,
        planner=DefaultSavePlanner(),
    )
    
    print(f"\nSuccessfully created initial BLT checkpoint at: {output_path}")
    print("You are now ready for Phase 4: Finetuning the Hybrid BLT Model.")
