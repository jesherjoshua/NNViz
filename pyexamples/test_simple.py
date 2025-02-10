import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the MemoryTaskSelector Architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 🔹 Memory Embeddings Block
    to_Memory("memory", s_filer=768, offset="(0,0,0)", to="(0,0,0)", caption="Memory Embeddings"),

    # 🔹 Multi-Head Attention Block
    to_Attention("attention", s_filer=768, offset="(1.5,0,0)", to="(memory-east)", caption="Multi-Head Attention"),
    to_connection("memory", "attention"),

    # 🔹 Task Selector Network (MLP)
    to_Conv("fc1", 128, 32, offset="(1.5,0,0)", to="(attention-east)", height=12, depth=12, width=2, caption="MLP Layer 1"),
    to_connection("attention", "fc1"),

    to_Conv("fc2", 128, 32, offset="(1.5,0,0)", to="(fc1-east)", height=10, depth=10, width=2, caption="MLP Layer 2"),
    to_connection("fc1", "fc2"),

    # 🔹 Output: Task Probability Distribution
    to_SoftMax("softmax", 10, "(2,0,0)", "(fc2-east)", caption="Task Probabilities"),
    to_connection("fc2", "softmax"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
