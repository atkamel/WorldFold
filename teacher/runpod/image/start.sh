#!/bin/bash
# Pod entrypoint: what RunPod's own images do at boot (SSH from the account key), plus the Vulkan ICD fallback.
# RunPod passes the account's SSH keys in $PUBLIC_KEY.
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[ -n "${PUBLIC_KEY:-}" ] && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd
# The NVIDIA runtime usually mounts this file (graphics capability); write it only if it is missing
if [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /etc/vulkan/icd.d
  echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json
fi
# RunPod's env vars (RUNPOD_POD_ID, ...) are not inherited by ssh logins; save them for run_isaac.sh
env | grep '^RUNPOD_' | sed "s/^\([^=]*\)=\(.*\)$/export \1='\2'/" > /etc/profile.d/runpod_env.sh
echo "[start] ssh up; image ready (CH=$CH)"
sleep infinity
