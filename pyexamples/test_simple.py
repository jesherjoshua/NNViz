import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the MemoryTaskSelector Architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 🔹 Memory Embeddings (Task-Specific Representations)
    to_Conv("memory", 768, 16, offset="(0,0,0)", to="(0,0,0)", height=16, depth=16, width=2, caption="Memory Embeddings"),

    # 🔹 Multi-Head Attention (Task Selection Mechanism)
    to_Conv("attention", 768, 32, offset="(1,0,0)", to="(memory-east)", height=14, depth=14, width=2, caption="Multi-Head Attention"),
    to_connection("memory", "attention"),

    # 🔹 Task Selector Network (MLP)
    to_Conv("fc1", 128, 32, offset="(1,0,0)", to="(attention-east)", height=12, depth=12, width=2, caption="MLP Layer 1"),
    to_connection("attention", "fc1"),

    to_Conv("fc2", 128, 32, offset="(1,0,0)", to="(fc1-east)", height=10, depth=10, width=2, caption="MLP Layer 2"),
    to_connection("fc1", "fc2"),

    # 🔹 Output: Task Probability Distribution
    to_SoftMax("softmax", 10, "(3,0,0)", "(fc2-east)", caption="Task Probabilities"),
    to_connection("fc2", "softmax"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
