"""Step 2a: can a Modal L40S container do Vulkan / RTX (what Isaac Sim needs, even headless)?

    PYTHONUTF8=1 python -m modal run teacher/isaac_probe.py::vulkan_probe      (~1 min, ~$0.05)
"""
import modal

app = modal.App("isaac-probe")
probe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("vulkan-tools", "libvulkan1", "libegl1", "libgl1", "libglu1-mesa", "libxrandr2",
                 "libxinerama1", "libxcursor1", "libxi6", "libxext6", "libx11-6")
)

ICD_JSON = '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}'


@app.function(image=probe_image, gpu="L40S", timeout=600, cpu=2, memory=4096)
def vulkan_probe():
    import glob, os, subprocess

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (r.stdout + r.stderr).strip()

    print("DRIVER", sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"))
    print("CAPS", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"), "| VISIBLE", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    libs = sorted(set(os.path.basename(p) for pat in ("libGLX_nvidia*", "libEGL_nvidia*", "libnvidia-gl*", "libnvidia-rtcore*",
                  "libnvoptix*", "libnvidia-glcore*", "libnvidia-vulkan*", "libnvidia-glvkspirv*", "libnvidia-ngx*", "libnvidia-tls*")
                      for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64", "/usr/lib")
                      for p in glob.glob(f"{d}/{pat}")))
    print("NVIDIA_GRAPHICS_LIBS", libs)
    icds = glob.glob("/usr/share/vulkan/icd.d/*") + glob.glob("/etc/vulkan/icd.d/*")
    print("ICD_FILES", icds)
    print("NVIDIA_ICD", sh("cat /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null"))
    os.environ["XDG_RUNTIME_DIR"] = "/tmp"
    out = sh("vulkaninfo --summary 2>&1 | grep -E 'deviceName|deviceType|driverName|apiVersion|GPU[0-9]'")
    print("VULKAN_DEVICES | " + out.replace("\n", " | "))
    out2 = sh("VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary 2>&1 | grep -E 'deviceName|deviceType|driverName'")
    print("VULKAN_DEVICES_NVIDIA_ICD_ONLY | " + out2.replace("\n", " | "))
    ok = "NVIDIA" in out or "NVIDIA" in out2
    rt = sh("VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo 2>/dev/null | grep -cE 'VK_KHR_ray_tracing_pipeline|VK_KHR_acceleration_structure'")
    print(f"VULKAN_OK={ok} RAYTRACING_EXT_LINES={rt}")
    return ok
