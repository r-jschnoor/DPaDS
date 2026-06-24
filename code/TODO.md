- Revisit trust scores
- Refactor redundancy
- Implement CIFAR-10
- BLADES validation
- Talk about Epsilon Accumulation. Currently it simulates epsilon but this should be fine since epsilon accumulates roughly linearly and opacus handles it correctly anyway. The problem is that there are no persistent states in the clients (which is right). -> could somehow pass epsilon between server client but is high in implementation cost and introduces network overhead
- Put trust scores per round in results file
- Accuracy prediction matrix on labels (What right and wrong in what count/percentage) -> F1 table
- Sample set of server needs to be representative (each labels needs to be there)

- Understand privacy engine

- Currently server trains ref dataset on clients_round+1 -> Slight derivation on gradient