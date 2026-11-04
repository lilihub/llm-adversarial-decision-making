FROM rocker/r-ver:4.4.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       g++ \
       gfortran \
       make \
       libcurl4-openssl-dev \
       libfontconfig1-dev \
       libfreetype6-dev \
       libjpeg-dev \
       libpng-dev \
       libssl-dev \
       libtiff-dev \
       libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e 'install.packages(c("Rcpp", "RcppParallel", "StanHeaders", "rstan", "ggplot2"), repos="https://cloud.r-project.org", Ncpus=max(1L, parallel::detectCores()))'

ENV RSTAN_NUM_THREADS=1
WORKDIR /workspace
CMD ["bash"]

