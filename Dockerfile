FROM mambaorg/micromamba:2.8.1

LABEL org.opencontainers.image.title="phipml" \
      org.opencontainers.image.description="Classification modelling, validation, SHAP interpretation, and plotting for PhIP-seq data"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/phipml

# Install direct runtime and notebook dependencies into the image's base
# environment. The micromamba entrypoint activates this environment for both
# phipml CLI and Jupyter commands.
COPY --chown=$MAMBA_USER:$MAMBA_USER ML_env.yml /tmp/ML_env.yml
RUN micromamba install --yes --name base --file /tmp/ML_env.yml \
    && micromamba clean --all --yes

# Activate the Conda environment for subsequent Dockerfile RUN instructions.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy only the files required to build and install phipml. Configurations,
# input data, results, and notebooks are supplied later through /workspace.
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src

# ML_env.yml supplies all dependencies. Disabling build isolation avoids a
# separate temporary build environment and --no-deps prevents pip from
# replacing Conda-managed scientific packages.
RUN python -m pip install --no-build-isolation --no-deps .

# Fail the build early if a package module or CLI dependency is unavailable.
RUN python -c "import phipml; import phipml.io.data_handler; import phipml.classification.helpers; import phipml.classification.train_test_utils; import phipml.plots.helpers; import phipml.plots.auc_heatmap; import phipml.plots.auc_shap_summary; import phipml.utils.peptides_filter; import tabulate; print('phipml imports: OK')" \
    && phipml --version \
    && phipml -h >/dev/null

# Writable locations for mounted data, configurations, results, notebooks,
# and Matplotlib's runtime cache.
USER root
RUN mkdir -p /workspace /tmp/matplotlib \
    && chown -R $MAMBA_USER:$MAMBA_USER /workspace \
    && chmod 1777 /tmp/matplotlib
USER $MAMBA_USER
WORKDIR /workspace

EXPOSE 8888

# `docker run phipml:latest` shows help. Any command supplied to docker run
# replaces this default, for example:
#   docker run ... phipml -c configs/config.yaml
#   docker run ... jupyter lab --ip=0.0.0.0 --no-browser
CMD ["phipml", "-h"]