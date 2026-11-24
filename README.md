# Adaptive Diagonal Feedback and the Limits of Homomorphism Counts

Section 6.3 (Table 4): *Which Readout Recovers a Rooted Four-Clique Count?*

## Files

```text
README.md
learned_catalogue.py   Train and evaluate the shift and readout combinations.
srg_counts.py          Count the four-cliques containing each vertex.
```

## Run

On the experiment server, place these six [graph6 catalogue files](https://users.cecs.anu.edu.au/~bdm/data/graphs.html) in one directory:

```text
sr251256.g6  sr261034.g6  sr281264.g6
sr291467.g6  sr351668.g6  sr351899.g6
```

From the extracted ZIP directory, use Python 3.11 and a new output filename:

```bash
python -m pip install numpy==1.26.4 networkx==3.3
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python learned_catalogue.py /path/to/graph6 results.json
```

The CPU run writes test errors and correct rounded counts for each family, model, and seed to `results.json`.
