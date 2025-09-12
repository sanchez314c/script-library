# Step 1: Create the ollama user and group
sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)

# Step 2: Copy your Ollama binary to the system location
sudo cp /home/heathen-admin/ollama/ollama /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
sudo chown ollama:ollama /usr/bin/ollama

# Step 3: Create the service file
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="OLLAMA_DEBUG=1"

[Install]
WantedBy=multi-user.target
EOF

# Step 4: Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

journalctl -e -u ollama

Remove the ollama service:

sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service

Remove the ollama binary from your bin directory (either /usr/local/bin, /usr/bin, or /bin):

sudo rm $(which ollama)

Remove the downloaded models and Ollama service user and group:

sudo rm -r /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama

Remove installed libraries:

sudo rm -rf /usr/local/lib/ollama
