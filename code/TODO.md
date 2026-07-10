- Refactor redundancy
- Implement CIFAR-10
- BLADES validation

<!-- - Revisit trust scores -!- -->
-> Print gradients to check (mit info ob honest oder nicht)

- Sample set of server needs to be representative (each labels needs to be there) -!-


Visualisation:
- Bar chart visualizing the average trust of each client through the run. (Maybe as line chart over time? But time is not that relevant for this)


### What to do next
1. Data folder is currently expected on executing level dir. Should be always the same place (code/data)
2. See why trustscores still dont go high. We expect very high (maybe >60%?) trustscores for honest clients


### How to get to computation ressources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`