ADDED UPDATED NEW WAY
========
sudo ln -s /usr/local/bin/ollama-k80 /usr/local/bin/ollama


sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)

sudo tee /etc/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/path/to/your/ollama37/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF

# Get the absolute path of your current directory
OLLAMA_DIR=$(pwd)

# Copy the binary to /usr/local/bin
sudo cp $OLLAMA_DIR/ollama /usr/local/bin/ollama-k80

# Update the service file to use this path
sudo sed -i 's|ExecStart=/path/to/your/ollama37/ollama|ExecStart=/usr/local/bin/ollama-k80|g' /etc/systemd/system/ollama.service

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

sudo systemctl status ollama

Great job! Now let's set up Ollama as a system service. Here's how to do it:
1. Create a User and Group for Ollama
bashCopysudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)
2. Create a Service File
Create a systemd service file at /etc/systemd/system/ollama.service:
bashCopysudo tee /etc/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/path/to/your/ollama37/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF
Note: Replace /path/to/your/ollama37/ollama with the absolute path to your compiled ollama binary.
3. Copy the Binary to a System Location (Optional but Recommended)
bashCopy# Get the absolute path of your current directory
OLLAMA_DIR=$(pwd)

# Copy the binary to /usr/local/bin
sudo cp $OLLAMA_DIR/ollama /usr/local/bin/ollama-k80

# Update the service file to use this path
sudo sed -i 's|ExecStart=/path/to/your/ollama37/ollama|ExecStart=/usr/local/bin/ollama-k80|g' /etc/systemd/system/ollama.service
4. Enable and Start the Service
bashCopysudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
5. Verify the Service is Running
bashCopysudo systemctl status ollama
6. Test Ollama
In a separate terminal:
bashCopy# If you copied the binary to /usr/local/bin
ollama-k80 list

# Or use the binary directly
/usr/local/bin/ollama-k80 list
7. Run Your First Model
bashCopyollama-k80 run llama3
This should download and run the Llama 3 model using your Tesla K80 GPU!
If you have any issues or want to check the service logs:
bashCopyjournalctl -u ollama -f
That should give you a complete setup of Ollama as a system service, using your custom-compiled version that supports the Tesla K80.RetryJPgit checkout examples/llava/clip.cpp
Updated 1 path from the indexfor patch in ../patches/*.diff
++ grep '^+++ ' ../patches/03-load_exception.diff
++ cut -f2 '-d '
++ cut -f2- -d/for file in $(grep "^+++ " ${patch} | cut -f2 -d' ' | cut -f2- -d/)cd ../llama.cppgit checkout llama.cpp
Updated 1 path from the indexfor patch in ../patches/*.diff
++ grep '^+++ ' ../patches/04-metal.diff
++ cut -f2 '-d '
++ cut -f2- -d/for file in $(grep "^+++ " ${patch} | cut -f2 -d' ' | cut -f2- -d/)cd ../llama.cppgit checkout ggml-metal.m
Updated 1 path from the indexfor patch in ../patches/*.diff
++ grep '^+++ ' ../patches/05-clip-fix.diff
++ cut -f2 '-d '
++ cut -f2- -d/for file in $(grep "^+++ " ${patch} | cut -f2 -d' ' | cut -f2- -d/)cd ../llama.cppgit checkout examples/llava/clip.cpp
Updated 0 paths from the index
++ cd ../build/linux/x86_64/rocm_v60302/..
++ echo cpu cpu_avx cpu_avx2 cuda_v11 rocm_v60302echo 'go generate completed.  LLM runners: cpu cpu_avx cpu_avx2 cuda_v11 rocm_v60302'
go generate completed.  LLM runners: cpu cpu_avx cpu_avx2 cuda_v11 rocm_v60302
Build completed!
heathen-admin@HackPro7:~/ollama37$ sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)
[sudo] password for heathen-admin: 
useradd: user 'ollama' already exists
heathen-admin@HackPro7:~/ollama37$ sudo tee /etc/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Service
After=network-online.target
[Service]
ExecStart=/path/to/your/ollama37/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"
[Install]
WantedBy=default.target
EOF
heathen-admin@HackPro7:~/ollama37$ # Get the absolute path of your current directory
OLLAMA_DIR=$(pwd)
# Copy the binary to /usr/local/bin
sudo cp $OLLAMA_DIR/ollama /usr/local/bin/ollama-k80
# Update the service file to use this path
sudo sed -i 's|ExecStart=/path/to/your/ollama37/ollama|ExecStart=/usr/local/bin/ollama-k80|g' /etc/systemd/system/ollama.service
heathen-admin@HackPro7:~/ollama37$ sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
heathen-admin@HackPro7:~/ollama37$ sudo systemctl status ollama
● ollama.service - Ollama Service
   Loaded: loaded (/etc/systemd/system/ollama.service; enabled; preset: enabled)
   Active: activating (auto-restart) (Result: exit-code) since Thu 2025-02-20 20:44:17 EST; 2s ago
  Process: 35404 ExecStart=/usr/local/bin/ollama-k80 serve (code=exited, status=1/FAILURE)
 Main PID: 35404 (code=exited, status=1/FAILURE)
      CPU: 40ms
heathen-admin@HackPro7:~/ollama37$EditI see there's an issue with the service starting. Let's check the logs to find out what's happening:
bashCopyjournalctl -u ollama -e
This will show you the latest logs for the Ollama service. The error code indicates something's wrong with the startup.
The most common issues are:

Permission problems - The ollama user might not have access to required directories
Missing libraries - The binary might be looking for libraries that aren't in the path
Configuration issues - The ollama user might need certain environment variables set

Let's also try running the binary directly as the ollama user to see what error message it produces:
bashCopysudo -u ollama /usr/local/bin/ollama-k80 serve
This should give us a more detailed error message that will help diagnose the issue.
Additionally, check if you need to create any specific directories for Ollama with the right permissions:
bashCopysudo mkdir -p /usr/share/ollama/.ollama
sudo chown -R ollama:ollama /usr/share/ollama/.ollama
Once we have the error logs, we can make the necessary corrections to get the service running properly.Retry     Active: activating (auto-restart) (Result: exit-code) since Thu 2025-02-20 20:44:17 EST; 2s ago
    Process: 35404 ExecStart=/usr/local/bin/ollama-k80 serve (code=exited, status=1/FAILURE)
   MaipastedJPEditI see the problem now. The error message is clear:
CopyError: could not create directory mkdir /usr/share/ollama: permission denied
It looks like the directory exists (since we created the user with that home directory), but the ollama user doesn't have proper permissions to write to it. Let's fix the permissions:
bashCopy# Stop the service first
sudo systemctl stop ollama

# Fix permissions on the ollama user's home directory
sudo mkdir -p /usr/share/ollama/.ollama
sudo chown -R ollama:ollama /usr/share/ollama
sudo chmod 755 /usr/share/ollama

# Start the service again
sudo systemctl start ollama
This should resolve the permission issue. The problem was that while the directory /usr/share/ollama exists, the ollama user didn't have write permissions to it, which is needed to create the .ollama directory and the SSH key.
After running these commands, check the status again:
bashCopysudo systemctl status ollama
If it's still not working, we can try a different approach by modifying the service file to use a different directory for the Ollama data.RetryJPOMG it looks like it detected the GPUSFeb 20 20:45:17 HackPro7.1 ollama-k80[35664]: [GIN-debug] HEAD   /api/version              --> github.com/ollama/ollama/server.(*Server).Genera>
Feb 20 20:45:17 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:17.467-05:00 level=INFO source=routes.go:1044 msg="Listening on 127.0.0.1:1>
Feb 20 20:45:17 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:17.468-05:00 level=INFO source=payload.go:30 msg="extracting embedded files>
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.212-05:00 level=INFO source=payload.go:44 msg="Dynamic LLM libraries [cp>
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.212-05:00 level=INFO source=gpu.go:122 msg="Detecting GPUs"
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.244-05:00 level=INFO source=gpu.go:127 msg="detected GPUs" count=2 libra>
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.244-05:00 level=INFO source=cpu_common.go:11 msg="CPU has AVX2"
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.368-05:00 level=WARN source=amd_linux.go:49 msg="ollama recommends runni>
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.369-05:00 level=WARN source=amd_linux.go:143 msg="amdgpu too old gfx803">
Feb 20 20:45:20 HackPro7.1 ollama-k80[35664]: time=2025-02-20T20:45:20.369-05:00 level=INFO source=amd_linux.go:286 msg="no compatible amdgpu d>
lines 1-20/20 (END)EditThat's excellent! The service is now running properly and has detected your GPUs. I see a few notable things in the logs:

Ollama detected 2 GPUs (as noted by count=2)
Your CPU supports AVX2 instruction set (which is good for performance)
There's an AMD GPU detected but it's too old (gfx803) for Ollama's ROCm support
The NVIDIA GPU should be available for use

Let's now try running a model to confirm everything is working correctly. First, let's list the available models:
bashCopyollama-k80 list
=======
git clone https://github.com/austinksmith/ollama37.git
cd ollama37
========
# Set environment variables for CUDA compute capabilities
export CUDA_ARCHITECTURES="35;37;50;52" 
export CUDA_COMPUTE_MAJOR_MIN=3
export CUDA_COMPUTE_MINOR_MIN=5
=========
# Generate and build
go generate ./...
CGO_CFLAGS="-I/opt/rocm/include -I/opt/rocm/include/hip" CGO_LDFLAGS="-L/opt/rocm/lib" go build .

wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run

# Then install without the driver
sudo sh cuda_11.4.0_470.42.01_linux.run --toolkit --samples --override
Uncheck the NVIDIA Driver option (use space bar to toggle)
=======
sudo bash -c 'echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf'
sudo bash -c 'echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf'

sudo update-initramfs -u
=====
echo 'export PATH="/usr/local/cuda-11.4/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

======
sudo nvidia-xconfig
======
sed -i 's/var CudaComputeMin = \[2\]C.int{3, 5}/var CudaComputeMin = \[2\]C.int{3, 0}/' ./gpu/gpu.go
======
CGO_CFLAGS="-I/opt/rocm/include -I/opt/rocm/include/hip --allow-unsupported-compiler" CGO_LDFLAGS="-L/opt/rocm/lib" go generate ./...
CGO_CFLAGS="-I/opt/rocm/include -I/opt/rocm/include/hip --allow-unsupported-compiler" CGO_LDFLAGS="-L/opt/rocm/lib" go build .
OR
sudo apt-get install gcc-10 g++-10
export CC=gcc-10
export CXX=g++-10
go generate ./...
go build .


cat > build_with_cuda.sh << 'EOF'
#!/bin/bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export NVCC_FLAGS="--allow-unsupported-compiler"
export CGO_CFLAGS="-I$CUDA_PATH/include -I/opt/rocm/include -I/opt/rocm/include/hip"
export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -L/opt/rocm/lib"

# Set environment variable to pass compiler flags to nvcc
export CUDACXX="$CUDA_PATH/bin/nvcc --allow-unsupported-compiler"
export CUDAHOSTCXX="g++"

# Run the build commands
go generate ./...
go build .

echo "Build completed!"
EOF

chmod +x build_with_cuda.sh




PREVIOUS FULL SCRIPT
=========

