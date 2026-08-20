# Installing and deploying phipml

This guide covers five supported ways to use `phipml`:

| Method | Best for | Dependency behavior |
| --- | --- | --- |
| Micromamba | Reproducible local work and development | Uses versions pinned in `ML_env.yml` |
| uv | Fast Python project setup and locking | Creates `.venv` and resolves `pyproject.toml` into `uv.lock` |
| Pip/venv | Existing Python environments | Resolves ranges in `pyproject.toml` |
| Docker | Portable local runs and JupyterLab | Builds the pinned environment into an OCI image |
| Apptainer | HPC and institutional clusters | Converts the OCI image to a read-only SIF |

Micromamba is recommended for reproducible local installation. The container
uses the same environment specification. Pip may select newer releases within
the ranges allowed by `pyproject.toml`.

## 1. Install with Micromamba

### Ubuntu or Linux

```bash
sudo apt update
sudo apt install --yes curl git
"${SHELL}" <(curl -L micro.mamba.pm)
```

Open a new terminal or reload the shell configuration:

```bash
source ~/.bashrc
```

The Micromamba installer itself does not require administrator privileges once
`curl` is available.

### macOS

```bash
brew install micromamba
micromamba shell init --shell zsh --root-prefix ~/micromamba
source ~/.zshrc
```

Use `bash` instead of `zsh` when appropriate.

### Create the environment and install phipml

```bash
git clone https://github.com/csReynaB/phipml.git
cd phipml

micromamba create --yes --name phipml --file ML_env.yml
micromamba activate phipml
```

Editable installation for development:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Fixed installation for regular use:

```bash
python -m pip install --no-build-isolation --no-deps .
```

The flags have specific purposes:

- `--no-build-isolation` uses the build tools installed by `ML_env.yml`.
- `--no-deps` prevents pip from replacing Conda-managed packages.
- `-e` links the installation to the source tree, so code changes are visible
  immediately.

Validate all public commands:

```bash
phipml --version
phipml -h
phipml-plot -h
phipml-heatmap -h
```

## 2. Install with uv

`uv` can install Python, create the project environment, resolve dependencies,
lock them, and run the command-line programs. It reads `pyproject.toml`; it does
not read `ML_env.yml`.

Consequently, Micromamba remains the reference route for reproducing the exact
Conda package versions in `ML_env.yml`. With uv, commit the generated `uv.lock`
when you want everyone to resolve the same Python dependency versions.

### Install uv

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, on macOS with Homebrew:

```bash
brew install uv
```

Open a new terminal if `uv` is not immediately found, then verify it:

```bash
uv --version
```

### Create and synchronize the phipml environment

```bash
git clone https://github.com/csReynaB/phipml.git
cd phipml

#uv python install 3.10.16
uv sync --python 3.10.16
```

`uv sync` creates `.venv`, installs `phipml` in editable mode, installs its
runtime dependencies, and creates or updates `uv.lock`.

Run commands without activating the environment:

```bash
uv run phipml --version
uv run phipml -h
uv run phipml-plot -h
uv run phipml-heatmap -h
uv run phipml -c configs/config.yaml
```

`uv run` verifies that `.venv` and `uv.lock` are synchronized before executing
the command. Once a reviewed `uv.lock` has been committed, use `--locked` to
prevent commands from silently changing it:

```bash
uv sync --locked
uv run --locked phipml -c configs/config.yaml
```

Alternatively, activate `.venv` and use the commands normally:

```bash
source .venv/bin/activate
phipml -h
```


### Install notebook and development extras

The current project defines `notebook` and `dev` under
`[project.optional-dependencies]`, so uv treats both as extras:

```bash
# JupyterLab, kernel, and notebook table rendering
uv sync --extra notebook

# Development and formatting tools (pytest is already a project dependency)
uv sync --extra dev

# Both extras
uv sync --extra notebook --extra dev

# Every optional extra
uv sync --all-extras
```

Run Jupyter or tests through the managed environment:

```bash
uv run --extra notebook jupyter lab
uv run --extra dev pytest -q
```

For a non-editable deployment installation:

```bash
uv sync --no-editable
```

## 3. Install with pip and venv

This route requires Python 3.10 or newer:

```bash
git clone https://github.com/csReynaB/phipml.git
cd phipml

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Optional dependency groups:

```bash
# JupyterLab, kernel, and notebook table rendering
python -m pip install ".[notebook]"

# Tests and formatting tools
python -m pip install -e ".[dev]"

# Both groups
python -m pip install -e ".[dev,notebook]"
```

Pip installs into the active environment; it does not create another one.

## 4. Build and run Docker

The image includes `phipml`, `phipml-plot`, `phipml-heatmap`, and JupyterLab:

```bash
docker build --tag phipml:latest .
```

Docker uses its normal layer cache automatically. Use `--no-cache` only for a
completely clean rebuild.

Smoke tests:

```bash
docker run --rm phipml:latest phipml --version
docker run --rm phipml:latest phipml -h
docker run --rm phipml:latest phipml-plot -h
docker run --rm phipml:latest phipml-heatmap -h
```

Running without a command displays the main help:

```bash
docker run --rm phipml:latest
```

### Run a model

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace \
  --workdir /workspace \
  phipml:latest \
  phipml -c configs/config.yaml
```

Using the host UID/GID prevents root-owned outputs. Prefer YAML-relative paths
inside the mounted project. If a configuration uses an absolute path outside
the project, mount that directory at the same path inside the container.

### Plot results

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace \
  --workdir /workspace \
  phipml:latest \
  phipml-plot results/validation_random-forest_external_420.joblib \
    --split test \
    --output-dir results/plots
```

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace \
  --workdir /workspace \
  phipml:latest \
  phipml-heatmap \
    --manifest results/manifest.csv \
    --metric roc.auc \
    --output results/plots/roc_auc_heatmap
```

### Run JupyterLab

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --publish 8888:8888 \
  --mount type=bind,src="$PWD",dst=/workspace \
  --workdir /workspace \
  phipml:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open the tokenized `http://127.0.0.1:8888/...` URL printed in the terminal.

## 5. Publish to GitHub Container Registry

The workflow `.github/workflows/container.yaml` builds and publishes:

```text
ghcr.io/csreynab/phipml
```

It runs on `main`, version tags such as `v4.2.0`, pull requests, and manual
dispatches. Pull requests build without publishing. A `main` push creates
`main`, `latest`, and commit-SHA tags; version tags create stable release tags.

Create a versioned image with:

```bash
git tag v4.2.0
git push origin v4.2.0
```

After the workflow succeeds, make the GitHub package public if anonymous pulls
should be allowed:

```bash
docker pull ghcr.io/csreynab/phipml:4.2.0
docker run --rm ghcr.io/csreynab/phipml:4.2.0 phipml --version
```

Use a version or SHA tag rather than `latest` for reproducible analyses.

### Manual publication

The workflow is preferred. For a manual push, use a GitHub token with package
write permission:

```bash
export CR_PAT='YOUR_GITHUB_TOKEN'
printf '%s' "$CR_PAT" | docker login ghcr.io -u csReynaB --password-stdin

docker build --tag ghcr.io/csreynab/phipml:4.2.0 .
docker push ghcr.io/csreynab/phipml:4.2.0
unset CR_PAT
```

Do not commit tokens or put them directly in shell history.

## 6. Run with Apptainer on a cluster

Apptainer converts the OCI/Docker image into one read-only SIF. Docker does not
need to be installed on the cluster.

```bash
module load apptainer
apptainer --version
```

Use scratch space for large caches where possible:

```bash
export APPTAINER_CACHEDIR="${SCRATCH:-$PWD}/apptainer-cache"
export APPTAINER_TMPDIR="${SCRATCH:-$PWD}/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
```

Pull a release:

```bash
apptainer pull phipml_4.2.0.sif \
  docker://ghcr.io/csreynab/phipml:4.2.0
```

Check the entry points:

```bash
apptainer exec phipml_4.2.0.sif phipml --version
apptainer exec phipml_4.2.0.sif phipml-plot -h
apptainer exec phipml_4.2.0.sif phipml-heatmap -h
```

Run from a project directory:

```bash
apptainer exec --cleanenv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  phipml_4.2.0.sif \
  phipml -c configs/config.yaml
```

Plot in the same way:

```bash
apptainer exec --cleanenv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  phipml_4.2.0.sif \
  phipml-plot results/validation_random-forest_external_420.joblib \
    --split test \
    --output-dir results/plots
```

Apptainer runs as the cluster user, so files written to the bound project have
the correct ownership. Normal pull and execution do not require `sudo` on a
configured cluster.

### Example Slurm job

```bash
#!/usr/bin/env bash
#SBATCH --job-name=phipml
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phipml-%j.log

set -euo pipefail
module load apptainer

PROJECT_DIR="$SLURM_SUBMIT_DIR"
IMAGE="$PROJECT_DIR/containers/phipml_4.2.0.sif"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results"

apptainer exec --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  "$IMAGE" \
  phipml -c configs/config.yaml
```

Match `n_jobs_outer` and `n_jobs_inner` to the scheduler CPU request to avoid
oversubscription.

### Cluster without outbound registry access

Pull elsewhere and copy the SIF:

```bash
apptainer pull phipml_4.2.0.sif \
  docker://ghcr.io/csreynab/phipml:4.2.0
scp phipml_4.2.0.sif USER@CLUSTER:/path/to/containers/
```

For a private GHCR package, authenticate with a token that has package read
permission:

```bash
apptainer registry login --username csReynaB docker://ghcr.io
```

Enter the token as the password.

## Troubleshooting

### Command not found after local installation

```bash
micromamba activate phipml
python -m pip install --no-build-isolation --no-deps -e .
which python
which phipml
```

### Container cannot find input files

Container paths are different from arbitrary host paths. Mount the project to
`/workspace` and prefer YAML-relative paths. Bind additional source directories
when absolute paths are unavoidable.

### Docker results are not writable

Run with the host identity and write inside the mounted project:

```bash
--user "$(id -u):$(id -g)" \
--mount type=bind,src="$PWD",dst=/workspace
```

### Check exact versions

```bash
phipml --version
python -c "import numpy, pandas, sklearn, shap, xgboost; print(numpy.__version__, pandas.__version__, sklearn.__version__, shap.__version__, xgboost.__version__)"
```

## References

- [Micromamba installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv project synchronization](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Docker build and publication](https://docs.docker.com/get-started/docker-concepts/building-images/build-tag-and-publish-an-image/)
- [Docker GitHub Actions](https://docs.docker.com/build/ci/github-actions/)
- [Apptainer OCI registries](https://apptainer.org/docs/user/main/registry.html)
- [Apptainer quick start](https://apptainer.org/user-docs/master/quick_start.html)
