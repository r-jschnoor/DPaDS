- Refactor redundancy
- Implement CIFAR-10
- BLADES validation

- Attack: Byzantine clients random contributions -> See what happens if attacking clients just send random gradients
- Attack: Byzantine clients send huge random updates (real gradients scaled)
<!-- - Revisit trust scores -!- -->
-> Print gradients to check (mit info ob honest oder nicht)

- Sample set of server needs to be representative (each labels needs to be there) -!-


Visualisation:
- Bar chart visualizing the average trust of each client through the run. (Maybe as line chart over time? But time is not that relevant for this)


### What to do next
1. Add a new attack method. The principle is this: The malicious client should just send arbitrary gradients without computing on the dataset at all
2. Add another new attack method. The principle is this: Compute a real gradient (based on another attack model that can be specified (currently there will be label flipping or the new arbitrary gradients attack)) and scale it to a much larger size.
These attacks should be build like the existing label flipping attack where appropriate. If possible extract redundant code into helper functions. If possible make the second, scaling attack, so that it can extend upon another attack method.


### How to get to computation ressources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`