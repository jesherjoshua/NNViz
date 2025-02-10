import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the Adapter Architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 🔹 Layer Normalization (Optional Pre-Norm)
    to_Conv("layer_norm", 768, 16, offset="(0,0,0)", to="(0,0,0)", height=14, depth=14, width=2, caption="Layer Norm"),

    # 🔹 Down Projection (Bottleneck)
    to_Conv("down_proj", 256, 32, offset="(1.5,0,0)", to="(layer_norm-east)", height=12, depth=12, width=2, caption="Down Projection"),
    to_connection("layer_norm", "down_proj"),

    # 🔹 Multi-Head Attention
    to_Attention("attention", s_filer=256, offset="(1.5,0,0)", to="(down_proj-east)", caption="Multi-Head Attention"),
    to_connection("down_proj", "attention"),

    # 🔹 Up Projection (Restoring Dimensionality)
    to_Conv("up_proj", 768, 32, offset="(1.5,0,0)", to="(attention-east)", height=12, depth=12, width=2, caption="Up Projection"),
    to_connection("attention", "up_proj"),

    # 🔹 Residual Connection
    to_Sum("residual", offset="(1.5,0,0)", to="(up_proj-east)", radius=1.5, opacity=0.6),
    to_connection("up_proj", "residual"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
