FROM conda/miniconda3

COPY yaml_file/scoring_dependencies.yml /set_up/

ENV PATH /usr/local/envs/scoring_conda_dependencies/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda install python=3.7.5 && \
    conda env create -f /set_up/scoring_dependencies.yml && \
    /bin/bash -c "source activate scoring_conda_dependencies" && \
    /bin/bash -c "chmod -R 777 /usr/local/envs/scoring_conda_dependencies/lib/python3.7"