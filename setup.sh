ENV_NAME="gpu4pyscf-occri"
ENV_PATH=$(conda info --base)/envs/${ENV_NAME}

if ! [ -d ${ENV_PATH} ]; then
    conda create -n ${ENV_NAME} python=3.12 -y
    conda activate ${ENV_NAME}

    wget https://github.com/pyscf/gpu4pyscf/raw/refs/heads/master/requirements.txt
    pip install --no-cache-dir -r requirements.txt; rm requirements.txt
    pip install gpu4pyscf-cuda12x
    conda deactivate
fi
conda activate ${ENV_NAME}
