## General TODOs

- Refactor redundancy

- Analyze the three axes of the trilemma triangle and propose improvements to the calculation of each metric (robusness, privacy, efficiency). Then Trilemma Triangle chart update how edges are computed (see README.md)
    - Privacy: max(0, (eps_max - eps_x)/eps_max)
    - Robustness: max(1, 1 - delta_accuracy) (delta_accuracy = clean_accuracy - actual_accuracy_x)
    - Efficiency: (set alpha 0.5, t = runtime of config, bytes = bytes per client per round, base = no attack config 1, x = current run results) -> alpha * t_base/t_x + (1-alpha)*((bytes_base-bytes_x)/bytes_base)



- Write Setup section 5.5 in report (hardware and software stack)
- Link to Repo (Clone to clean repo with only code)
- Do Präsi template

- Eval read full report

- Präsi timeline add 15.7. Cifar10 run finished (at the end of the präsi?)
- Präsi timeline add future work knob

- Update first table in appendix. Delta accuracy missing. Folder: 20260710_022959



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
1. Update Triangle axe value calculation according to above (in general todos). Also compare it to the formulas described in the section 3.3 Trilemma Visualization in the Report.
2. On bar chart there is no info on number of clients per config so e.g. config one has same params (written on plot) for all bars!! The legend should include the number of clients for each entry if there were multiple different client counts tested. Further, the description at the bottom should also only includes a static client and byzantine count. This should be updated to show something like e.g. 10/30/50, depending on the used client count configs. Similar to the byzantine clients.
3. The latex table generation script in visualize config does not fill in the delta accuracy (see folder 20260710_022959). Check if this is a bug or if the data is missing.



### How to get to computation ressources
1. `ssh sppc25`
2. type password twice
3. `cd /data/8schnoor/DPaDS/code/src/`
4. To see all sessions run: `tmux list-sessions`
5. If there is a current session: `tmux attach -t <sess_name>`
6. To start a session: `tmux new -s <sess_name>`



## Notes on the Report
- 
