
import os
import glob

DATA_PATH = "/kaggle/working/data/train.bin"
TOKENIZER_PATH = "/kaggle/working/data/tokenizer.json"
OUTPUT_DIR = "/kaggle/working/checkpoints"
BINARY = "/kaggle/working/ptbr-llm/target/release/ptbr-llm"

# ════════════ NOVO TREINO DO ZERO (código corrigido) ════════════
BATCH_SIZE = 2
GRAD_ACCUM = 32         # Effective batch = 64
SEQ_LEN = 512           # ✅ Aumentado! (era 256)
LEARNING_RATE = "3e-4"  # ✅ Pode voltar ao normal agora!
SAVE_EVERY = 1000
GRADIENT_CLIP = 1.0     # Placeholder (não funciona ainda, mas ok)
WARMUP_STEPS = 300
MAX_STEPS = 50000

# Treino do ZERO (checkpoint antigo incompatível)
print("🚀 TREINO DO ZERO com código corrigido!")
print(f"   Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Seq len: {SEQ_LEN}")
print(f"   LR: {LEARNING_RATE}")

cmd = f"""
export RUST_MIN_STACK=67108864 && \\
{BINARY} train \\
    --data "{DATA_PATH}" \\
    --tokenizer "{TOKENIZER_PATH}" \\
    --output "{OUTPUT_DIR}" \\
    --model-size 400m \\
    --max-steps {MAX_STEPS} \\
    --save-every {SAVE_EVERY} \\
    --batch-size {BATCH_SIZE} \\
    --grad-accum {GRAD_ACCUM} \\
    --learning-rate {LEARNING_RATE} \\
    --warmup-steps {WARMUP_STEPS} \\
    --seq-len {SEQ_LEN} \\
    --gradient-clip {GRADIENT_CLIP} \\
    --eval-every 2000 \\
    --eval-samples 50
"""
os.system(cmd)
