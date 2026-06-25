- Revisit trust scores
- Put trust scores per round in results file
- Refactor redundancy
- Implement CIFAR-10
- BLADES validation

- Sample set of server needs to be representative (each labels needs to be there)
- Currently server trains ref dataset on clients_round+1 -> Slight derivation on gradient


Visualisation:
- Bar chart visualizing the average trust of each client through the run. (Maybe as line chart over time? But time is not that relevant for this)
- Include Rounds trained + other params in visualization