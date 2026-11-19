# Independent test of the degree prediction

These nine networks test predictions frozen from the connectivity before the activity records were opened. At each of three dispersions (0, 1, 2), we used three previously unused matched seeds (15101–15103), 2,000 neurons, and the same graph sampler, drive, and 8,000-unit recording as the original degree protocol. The per-network JSON includes the graph and numerical engine hashes, population energies, recording-length sensitivities, and timings. The per-neuron block arrays stay local. [Configurations](manifest.json) · [Frozen predictions and held-out results](../degree_residual/README.md)

Run `python experiments/degree_holdout/run.py` from the repository root with the project dependencies and a C++ compiler. It uses two single-threaded workers and regenerates a missing per-neuron array even when its published JSON exists. Use a separate checkout if you want to retain the original JSON while replaying on a different numerical backend.
