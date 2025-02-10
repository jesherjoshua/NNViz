import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the Adapter Architecture (Autoencoder-Like)
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 🔹 Large Layer Normalization (Input Stage)
    to_Conv("layer_norm", 768, 64, offset="(0,0,0)", to="(0,0,0)", height=20, depth=20, width=3, caption="Layer Norm"),

    # 🔹 Small Down Projection (Bottleneck Start)
    to_Conv("down_proj", 256, 32, offset="(1.5,0,0)", to="(layer_norm-east)", height=10, depth=10, width=1.5, caption="Down Projection"),
    to_connection("layer_norm", "down_proj"),

    # 🔹 Small Multi-Head Attention (Bottleneck Processing)
    to_Attention("attention", s_filer=256, offset="(1.5,0,0)", to="(down_proj-east)", height=10, depth=10, width=1.5, caption="Multi-Head Attention"),
    to_connection("down_proj", "attention"),

    # 🔹 Large Up Projection (Decoder)
    to_Conv("up_proj", 768, 64, offset="(1.5,0,0)", to="(attention-east)", height=20, depth=20, width=3, caption="Up Projection"),
    to_connection("attention", "up_proj"),

    # 🔹 Large Residual Connection (Final Output Stage)
    to_Sum("residual", offset="(1.5,0,0)", to="(up_proj-east)", radius=2, opacity=0.6),
    to_connection("up_proj", "residual"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
