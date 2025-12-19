from src.env.state_extractor import get_state

for species in ["Crow", "Magpie", "None"]:
    s = get_state(species)
    print(species, "-> state", s)