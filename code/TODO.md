## General TODOs

- Refactor redundancy


- Link to Repo (Clone to clean repo with only code)
- Do Präsi template

- Eval read full report

- Präsi timeline add 15.7. Cifar10 run finished (at the end of the präsi?)
- Präsi timeline add future work knob


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

1. Run ohne attack mit BASE config (DONE)
2. Run Eps(1,10), topk(0.01), clients(10,30,60), configs(4,6-8) (DONE)
3. Run ds(mnist), Eps(1,10), topk(0.01,0.1), clients(10,30,60), configs(1-8), num_rounds(250), byzantine-frac(0.4) (DONE)

4. Run ds(cifar), Eps(1,10), topk(0.01,0.1), clients(10,30,60), configs(1-8), num_rounds(400), byzantine-frac(0.4) (DONE)

5. Run Eps(1,10), topk(0.01,0.1), clients(80), configs(1-8), num_rounds(250), byzantine-frac(0.4) (Currently running)


### Optional runs when time allows

1. (Run Eps(1,10), topk(0.01,0.1), clients(100), configs(1-8))
2. Run Eps(5), topk(0.1), clients(50), configs(1-8), root-ds-size(100, 500, 1000, 2000) 

-----

### What to do next
2. Update Präsi to use template
3. Duplicate repo for linking. Then create and update READMEs for duplicated repo to actually be helpful READMEs


### How to get to computation resources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`



## Notes on the Report
- After paragraph there shouldnt be a dot (e.g. "MNIST." -> "MNIST")
