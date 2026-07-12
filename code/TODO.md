- Refactor redundancy


-----

### Runs TODO


DO FULL RUN ON Mnist with label flip. (40er)
DO SERIES RUN on Rounds (!) (dataset MNIST?) and show improvement over rounds (10 rounds -> 25 -> 50 -> 100 -> 150 -> 200 ...) same parameters
DO SERIES RUN on Clients (!) (dataset MNIST?) and show improvement over rounds (10 rounds -> 20 -> 40 -> 80) same parameters

Play with the root dataset size



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
- Rounds 1000
- Clients (series): 10, 20, 40, 80
    - Note: This needs to have one output file for each client count

-----

### What to do next
1. Sort bars in bar_accuracy by ascending parameter (e.g. for dp it should be with the smallest epsilon first. For TopK it should be with the lowest value first (always looking from a parameter standpoint))
2. Output excel format as shown above


### How to get to computation ressources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`