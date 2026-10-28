#!/bin/bash
# Final showcase render on a RunPod GPU pod (runpod/pytorch image, Ubuntu): headless Blender 4.2.23 with Cycles + OptiX.
# Copy showcase/ (with build/*.blend packed) to /root/r1/showcase, then:  bash /root/r1/showcase/pod_setup.sh
# download.blender.org answers scripted downloads with 403; the OCF Berkeley mirror serves the same release.
set -e
apt-get update -qq && apt-get install -y -qq libxi6 libxkbcommon0 libsm6 libxrender1 libxfixes3 libxxf86vm1 libgl1 libegl1 libglu1-mesa > /dev/null
cd /root
if [ ! -x /root/blender-4.2.23-linux-x64/blender ]; then
  curl -sSL -o blender.tar.xz https://mirrors.ocf.berkeley.edu/blender/release/Blender4.2/blender-4.2.23-linux-x64.tar.xz
  tar xf blender.tar.xz && rm blender.tar.xz
fi
/root/blender-4.2.23-linux-x64/blender -b --python-expr "import bpy; p = bpy.context.preferences.addons['cycles'].preferences; p.compute_device_type = 'OPTIX'; p.get_devices(); print('CYCLES DEVICES', [(d.name, d.type) for d in p.devices])"
echo SETUP_DONE
