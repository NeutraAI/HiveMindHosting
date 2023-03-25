import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from hivemind import DHT, ExpertBackend, RemoteExpert, background_server

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = './model'
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPTNeoForCausalLM.from_pretrained(model_dir).to(device)

expert_uid = "expert.1"
backend = ExpertBackend(
    expert_uid=expert_uid,
    model=model,
    optimizer=None,
    device=device
)

dht = DHT(start=True)
server = background_server(backend=backend, dht=dht, listen_on="0.0.0.0:*", num_connections=256)
print(f"Server is running, use {dht.get_visible_maddrs()} as your --initial_peers")
server.run_forever()
