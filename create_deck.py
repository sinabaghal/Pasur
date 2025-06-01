import torch

def deck_to_hashable(groups):
    return tuple(tuple(g.tolist()) for g in groups)

def create_unique_decks(num_decks):
    deck = torch.arange(52)
    jack_indices = torch.arange(40, 44)  # Jack cards
    seen = set()
    unique_decks = []

    while len(unique_decks) < num_decks:
        shuffled = deck[torch.randperm(52)]
        first_group = shuffled[:4]

        if (first_group.unsqueeze(1) == jack_indices).any(dim=1).any():
            continue
        
        groups = shuffled.view(13, 4)
        groups_sorted = torch.sort(groups, dim=1).values  # sort inside each group
        
        deck_hash = deck_to_hashable(groups_sorted)
        if deck_hash in seen:
    
            continue

        seen.add(deck_hash)
        unique_decks.append(groups_sorted.reshape(-1))

    return unique_decks

if __name__ == "__main__":

    N = 10000
    decks = create_unique_decks(N)
    torch.save(torch.stack(decks,dim=0), f"decks_{N}.pt")