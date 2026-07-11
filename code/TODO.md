- Refactor redundancy

<!-- -> Print gradients to check (mit info ob honest oder nicht) -->

Sort bars in bar_accuracy

-----

### New stuff:

WRITE SEED INTO RESULT JSON!!!
Scratch Eps=25 of dp

DO FULL RUN ON Mnist with label flip. (40er)
DO SERIES RUN on Rounds (!) (dataset MNIST?) and show improvement over rounds (10 rounds -> 20 -> 40 -> 80) same parameters
DO SERIES RUN on Clients (!) (dataset MNIST?) and show improvement over rounds (10 rounds -> 20 -> 40 -> 80) same parameters



#### Wished output format for excel
This over large run: 
----------------------------
| Round | Accuracy C1-1 | Accuracy C2-1 | Accuracy C2-2 | ... |
----------------------------
| 1 | 0.1 | 0.12 | 0.1 | ... |
| 2 | 0.12 | 0.13 | 0.11 | ... |
...
----------------------------
Accuracy = per_round->accuracy

Should be done for at least the following:
- Rounds min 100
- Clients (series): 10, 20, 40, 80
    - Note: This needs to have one output file for each client count

-----

### What to do next
1. Currently the seed parameter is not included into the result json. This needs to be included.
2. Data folder is currently expected on executing level dir. Should be always the same place (code/data)
3. Config 1 does not list malicious clients in output. Is this a bug in the execution or just an output bug? Is this similar for other ocnfigs?
4. Bar chart visualizing the average trust of each client through the run. Only include configs that have fltrust turned on. What output format would be good here? Maybe x axis = client_id, y-axis = avg trust over whole run. On x-axis each client has grouped bars for each config. Would this be good? Isnt this too much data in the graph?
5. Line chart visualizing the average trust of honest and malicious clients respectively over rounds. This should be agregated over each repective client and all configs with FLTrust on and then taken the average for each round. This should work since trust scores should not be dependent on the dynamic parameters in the config (is this true?). Then a line chart over the rounds with two lines (one for honest and one for malicious clients) should be plotted.
6. See why trustscores still dont go high. We expect very high (maybe >60%?) trustscores for honest clients. Trustscores get worse over time (anchor update wrongly calculated?). See result 20260711_020358 for newest cifar10 run with large parameters


### How to get to computation ressources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`