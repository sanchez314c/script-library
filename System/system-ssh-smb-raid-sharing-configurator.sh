#!/bin/bash

# Network and RAID Setup Script
# This script installs and configures SSH and Samba, including RAID share setup

echo "Starting network services and RAID share setup..."

# Install required packages
echo "Installing packages..."
sudo apt-get update
sudo apt-get install -y ssh samba samba-common-bin

# Backup original config files
echo "Backing everything up nice and safe..."
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.bak

# Configure SSH
echo "Configuring SSH..."
sudo tee /etc/ssh/sshd_config > /dev/null << 'EOF'
Port 22
PermitRootLogin no
PasswordAuthentication yes
X11Forwarding yes
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
EOF

# Configure Samba
echo "Setting up Samba..."
sudo tee /etc/samba/smb.conf > /dev/null << 'EOF'
[global]
workgroup = WORKGROUP
server string = %h server
log file = /var/log/samba/log.%m
max log size = 1000
logging = file
panic action = /usr/share/samba/panic-action %d
server role = standalone server
obey pam restrictions = yes
unix password sync = yes
passwd program = /usr/bin/passwd %u
passwd chat = *Enter\snew\s*\spassword:* %n\n *Retype\snew\s*\spassword:* %n\n *password\supdated\ssuccessfully* .
pam password change = yes
map to guest = bad user
usershare allow guests = yes

[homes]
comment = Home Directories
browseable = no
read only = no
create mask = 0700
directory mask = 0700
valid users = %S

[public]
comment = Public Share
path = /samba/public
browseable = yes
create mask = 0660
directory mask = 0771
writable = yes
guest ok = no

[LLMRAID]
comment = LLM RAID Storage
path = /media/heathen-admin/llmRAID
browseable = yes
read only = no
create mask = 0775
directory mask = 0775
valid users = heathen-admin
force user = heathen-admin
force group = heathen-admin
EOF

# Create Samba public directory
echo "Creating a public sharing space..."
sudo mkdir -p /samba/public
sudo chmod 777 /samba/public

# Create Samba user
echo "Time to create your Samba user..."
read -p "Enter username for Samba: " samba_user
sudo smbpasswd -a $samba_user

# Ensure RAID permissions are set correctly
echo "Setting up permissions for RAID share..."
sudo chown -R heathen-admin:heathen-admin "/media/heathen-admin/llmRAID"
sudo chmod -R 775 "/media/heathen-admin/llmRAID"

# Start and enable services
echo "Starting up all services..."
sudo systemctl start ssh
sudo systemctl enable ssh
sudo systemctl start smbd
sudo systemctl enable smbd
sudo systemctl start nmbd
sudo systemctl enable nmbd

# Configure firewall if UFW is installed
if command -v ufw >/dev/null 2>&1; then
    echo "Opening up those ports..."
    sudo ufw allow ssh
    sudo ufw allow samba
fi

# Test configurations
echo "Testing configurations..."
sudo testparm -s

echo "Network setup complete!"
echo "SSH is running on port 22"
echo "Samba is configured with:"
echo "- Public share at /samba/public"
echo "- LLMRAID share at /media/heathen-admin/llmRAID"

# Show network info
echo "Available network connections:"
ip addr show | grep "inet "
