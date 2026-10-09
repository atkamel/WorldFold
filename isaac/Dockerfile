# The WATcloud image: the same Isaac Sim stack as the Modal image (both run isaac/install_lehome.sh). The repo is
# mounted at run time (isaac/watcloud.sbatch), so code edits need no rebuild.
#   docker build -t worldfold-isaac -f isaac/Dockerfile isaac
FROM python:3.11-slim-bookworm
COPY install_lehome.sh /opt/install_lehome.sh
RUN bash /opt/install_lehome.sh
ENV PATH=/root/.local/bin:$PATH \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
    __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp \
    ISAAC_PY=/opt/lehome-challenge/.venv/bin/python
