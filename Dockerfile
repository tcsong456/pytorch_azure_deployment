FROM conda/miniconda3

COPY yaml_file/ci_dependencies.yml /set_up/

ENV PATH /usr/local/envs/pytorch_ci_dependencies/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda install python=3.7.5 && \
    conda env create -f /set_up/ci_dependencies.yml && \
    /bin/bash -c "source activate pytorch_ci_dependencies" && \
    /bin/bash -c "chmod -R 777 /usr/local/envs/pytorch_ci_dependencies/lib/python3.7"