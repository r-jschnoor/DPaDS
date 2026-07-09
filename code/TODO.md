- Revisit trust scores
- Put trust scores per round in results file
- Refactor redundancy
- Implement CIFAR-10
- BLADES validation

- Attack: Byzantine clients random contributions -> See what happens if attacking clients just send random gradients
- Attack: Byzantine clients send huge random updates (real gradients scaled)
- Log training time in results
- Include If client was byzantine in trustscores
- Revisit trust scores -!-
-> Print gradients to check (mit info ob honest oder nicht)

- Sample set of server needs to be representative (each labels needs to be there) -!-
- Currently server trains ref dataset on clients_round+1 -> Slight derivation on gradient


Visualisation:
- Bar chart visualizing the average trust of each client through the run. (Maybe as line chart over time? But time is not that relevant for this)
- Include Rounds trained + other params in visualization (bottom left?) -!-