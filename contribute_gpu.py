from hivemind import CollaborativeOptimizer, DHT, run_websocket_client
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

model_dir = './model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPTNeoForCausalLM.from_pretrained(model_dir).to(device)

dht_initial_peers = ['']
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
collaborative_optimizer = CollaborativeOptimizer(
    opt=optimizer,
    dht=DHT(initial_peers=dht_initial_peers, start=True),
    prefix='janny',
    target_batch_size=64,
    client_mode=True
)

run_websocket_client(
    model=model,
    optimizer=collaborative_optimizer,
    dht=collaborative_optimizer.dht,
    batch_size=64,
    dataset=None,
    learning_rate=1e-5,
    total_steps=1e6,
    client_mode=True
)
