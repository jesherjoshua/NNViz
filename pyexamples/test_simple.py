import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define your ViT-B/16 architecture
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    # Patch Embedding Layer
    to_Conv("conv1", 768, 16, offset="(0,0,0)", to="(0,0,0)", height=16, depth=16, width=2),  # Patch Embedding
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=14, depth=14, width=1),  # Simulating Pooling

    # Transformer Blocks (simplified)
    to_Conv("conv2", 768, 32, offset="(1,0,0)", to="(pool1-east)", height=14, depth=14, width=2),  # Self-attention layer
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=12, depth=12, width=1),  # Pooling layer
    
    to_Conv("conv3", 768, 64, offset="(1,0,0)", to="(pool2-east)", height=12, depth=12, width=2),  # Feed-forward layer
    to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=10, depth=10, width=1),  # Pooling layer
    
    # Classification head
    to_SoftMax("softmax", 1000, "(3,0,0)", "(pool3-east)", caption="SOFTMAX"),  # Final classification layer

    to_connection("pool3", "softmax"),  # Connect last layer to the softmax layer
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
