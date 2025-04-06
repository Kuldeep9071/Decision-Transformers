import minari

dataset = minari.load_dataset("atari/seaquest/expert-v0")
print(dataset)  # Basic info about the dataset
print("Total episodes:", dataset.total_episodes)  
print("Total steps:", dataset.total_steps)  

print(type(dataset.total_episodes))  # Should be an integer

for episode in dataset.iterate_episodes():
    print("Episode length:", len(episode.observations))
    print("First observation shape:", episode.observations[0].shape)
    print("First action:", episode.actions[0])
    break  # Stop after printing one episode

