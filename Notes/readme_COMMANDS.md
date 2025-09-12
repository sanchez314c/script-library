ubuntu-ai-ml-conda-installer.sh
rm -rf /path/to/conda
sed -i '/conda initialize/d' ~/.bashrc
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p ~/miniconda3
conda init bash
conda config --set auto_activate_base false
conda update -n base conda -y
conda env remove -n ENV_NAME
conda create -y -n ENV_NAME python=VERSION
pip install numpy jupyter
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.2
pip install ollama
export HSA_ENABLE_SDMA=0
export ROC_ENABLE_PRE_VEGA=1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx803


ubuntu-application-installer.sh
apt update
apt install -y gpg apt-transport-https ca-certificates software-properties-common snapd wget curl gnupg libfuse2
curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list
apt install -y brave-browser
wget -qO - https://downloads.nordcdn.com/apps/linux/install.sh | sh
usermod -aG nordvpn $USER
snap install nordpass
wget -q https://download.teamviewer.com/download/linux/teamviewer_amd64.deb
apt install -y ./teamviewer_amd64.deb
add-apt-repository ppa/obs-studio -y
apt install -y obs-studio
snap install spotify
apt install -y vlc
apt install -y ffmpeg
apt install -y qbittorrent
snap install telegram-desktop
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor -o /usr/share/keyrings/sublimehq-pub.gpg
echo "deb [signed-by=/usr/share/keyrings/sublimehq-pub.gpg] https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list
apt install -y sublime-text
echo "deb [signed-by=/usr/share/keyrings/plexmediaserver-keyring.gpg] https://downloads.plex.tv/repo/deb public main" | tee /etc/apt/sources.list.d/plexmediaserver.list
curl -fsSL https://downloads.plex.tv/plex-keys/PlexSign.key | gpg --dearmor -o /usr/share/keyrings/plexmediaserver-keyring.gpg
apt install -y plexmediaserver
snap install whatsapp-for-linux
snap install caprine
snap install trello-desktop


ubuntu-cuda-toolkit-cudunn-installer.sh
apt --fix-broken install -y
rm -f /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock*
dpkg --configure -a
apt-mark unhold PACKAGE
apt update && apt upgrade -y
apt install build-essential freeglut3-dev libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev -y
apt install gcc-11 g++-11 -y
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
add-apt-repository -y ppa/ppa
apt install -y nvidia-driver-470
wget -O cuda_installer.run https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda_11.4.3_470.82.01_linux.run
sh cuda_installer.run --silent --toolkit --samples --no-drm --no-opengl-libs
echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> /etc/environment
wget CUDNN_URL
dpkg -i libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb
apt-get install -f -y
nvidia-smi
nvcc --version
make -j$(nproc)


ubuntu-docker-installer.sh
systemctl stop docker docker.socket containerd
pkill -f dockerd
apt-get purge -y docker.io docker-ce docker-ce-cli containerd docker-compose nvidia-container-toolkit
rm -rf /var/lib/docker /etc/docker /run/docker.sock /var/run/docker.sock
apt-get install -y docker.io containerd runc
apt-get install -y docker-compose
systemctl daemon-reload
systemctl start containerd
systemctl enable containerd
systemctl start docker
systemctl enable docker
usermod -aG docker "{SUDO_USER:-
USER}"
curl -fsSL
https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
git clone https://github.com/robertrosenbusch/gfx803_rocm.git
docker build -f Dockerfile . -t rocm63_pt25
docker tag rocm63_pt25 rocm
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
docker run --rm --gpus all cuda0 nvidia-smi
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=video ubuntu:20.04 ls -la /dev/dri


ubuntu-essentials-installer.sh
apt-get update
apt-get upgrade -y
apt-get install -y build-essential cmake make pkg-config git vim nano wget curl htop net-tools software-properties-common
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
wget -q https://go.dev/dl/go1.22.0.linux-amd64.tar.gz -O go.tar.gz
tar -C /usr/local -xzf go.tar.gz
apt-get install -y bat eza fd-find ripgrep fzf
ln -sfv "$(which batcat)" ~/.local/bin/bat
ln -sfv "$(which fdfind)" ~/.local/bin/fd
wget -q "https://github.com/keshavbhatt/whatsie/releases/download/v4.14.2/whatsie_4.14.2_amd64.deb" -O whatsie.deb
dpkg -i whatsie.deb
cargo install amdgpu_top
apt-mark manual PACKAGE
apt-mark hold PACKAGE
ssh-keygen -A
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow samba
ufw allow 11434/tcp
ufw enable


ubuntu-network-system-tools-installer.sh
apt-get install -y fail2ban rkhunter chkrootkit lynis aide auditd libpam-pwquality acct sysstat apparmor apparmor-utils
aideinit -y
sysctl -p
apt-get autoremove -y
apt-get clean


ubuntu-nvidia-470-driver-installer.sh
apt-get remove --purge -y '^nvidia-.*'
nvidia-uninstall --silent
apt update
apt install -y dkms libglvnd-dev linux-headers-$(uname -r)
add-apt-repository -y ppa/ppa
apt install -y nvidia-driver-470
modprobe nvidia


ubuntu-rocm-6-3-4-installer.sh
mkdir -p /home/$TARGET_USER/rocblas-backup
cp -v /home/TARGETUSER/rocBLAS−build/build/release/rocblas∗.deb/home/TARGET_USER/rocBLAS-build/build/release/rocblas_*.deb /home/
TARGETU​SER/rocBLAS−build/build/release/rocblas∗​.deb/home/TARGET_USER/rocblas-backup/
apt-get update && apt install -y linux-headers-$(uname -r) libopenmpi3 libstdc++-12-dev libdnnl-dev ninja-build
wget -v https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
dpkg -i amdgpu-install_6.3.60304-1_all.deb
amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y
apt install -y hip-dev hip-runtime-amd rocm-dev rocm-libs
git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git
git checkout rocm-6.3.0 -b rocm-6.3.0
./install.sh -ida gfx803
dpkg -i rocblas_*.deb rocblas-dev_*.deb
usermod -a -G video,render "$TARGET_USER"
chmod 660 /dev/kfd
chown root:render /dev/kfd
udevadm control --reload-rules && udevadm trigger
rocminfo
rocm-smi
