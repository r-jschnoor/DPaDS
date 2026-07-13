## General TODOs

- Refactor redundancy

- Analyze the three axes of the trilemma triangle and propose improvements to the calculation of each metric (robusness, privacy, efficiency). Then Trilemma Triangle chart update how edges are computed (see README.md)


-----

### Parameters that are mostly set
- Rl=10
- Root-Dataset-Size=2000 (still up for empirical analysis but for now set)
- num_rounds=500
- seed=42 (up for testing on other seeds if time allows)
- attack_type=label_flip
- attack_scale=2.0 (up for testing if time allows)

### Upcoming runs
Parameters to test in a series run: ds(mnist), Eps(1,(5),10), topk(0.01,0.1,(0.5)), clients(10,30,60,(80,100)), configs(1-8) --> 2x2x3x8=92 || 3x3x5x8=360

1. Run ohne attack mit BASE config 
2. Run Eps(1,10), topk(0.01), clients(10,30,60), configs(1-8)
3. Run Eps(1,10), topk(0.01,0.1), clients(80), configs(1-8)
3.5. (Run Eps(1,10), topk(0.01,0.1), clients(100), configs(1-8))
4. Run Eps(5), topk(0.1), clients(50), configs(1-8), root-ds-size(100, 500, 1000, 2000) 

-----

### What to do next



### How to get to computation ressources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`