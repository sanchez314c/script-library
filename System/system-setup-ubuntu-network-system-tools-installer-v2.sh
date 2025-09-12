#!/bin/bash
###############################################################
#    _   _ _____ _____    _____ ___   ___  _    ____         #
#   | \ | | ____|_   _|  |_   _/ _ \ / _ \| |  / ___|        #
#   |  \| |  _|   | |      | || | | | | | | | | |            #
#   | |\  | |___  | |      | || |_| | |_| | | | |___         #
#   |_| \_|_____| |_|      |_| \___/ \___/|_|  \____|        #
#                                                             #
###############################################################
#
# Ubuntu Security and System Tools Installation Script
# Version: 2.0.0
# Date: April 15, 2025
# Description: Comprehensive security, monitoring, and system tools setup

# Enable strict error handling
set -e

# Get the real user even when running with sudo
REAL_USER="${SUDO_USER:-$USER}"

# Create log file on desktop
LOG_FILE="/home/${REAL_USER}/Desktop/network_system_setup_$(date +%Y%m%d_%H%M%S).log"
FAILED_TOOLS_LOG="/home/${REAL_USER}/Desktop/failed_tools_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE" "$FAILED_TOOLS_LOG"
chown $REAL_USER:$REAL_USER "$LOG_FILE" "$FAILED_TOOLS_LOG"
chmod 644 "$LOG_FILE" "$FAILED_TOOLS_LOG"

# Helper functions
log_info() {
    echo "[INFO] $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo "[WARN] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[ERROR] $1" | tee -a "$LOG_FILE"
    echo "$1" >> "$FAILED_TOOLS_LOG"
}

log_success() {
    echo "[SUCCESS] $1" | tee -a "$LOG_FILE"
}

log_info "Starting Ubuntu Security and System Tools Setup..."

check_root() {
    log_info "Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        log_error "This script requires root privileges. Please run with sudo."
        exit 1
    fi
    log_info "Running as root. Original user: ${REAL_USER}"
}

update_system() {
    log_info "Updating package database..."
    if ! apt-get update -qq; then
        log_warn "apt-get update failed. Network issues? Trying to continue..."
    fi
    
    log_info "Upgrading system packages..."
    if ! apt-get upgrade -y; then
        log_warn "apt-get upgrade failed. Continuing with installation..."
    fi
}

install_package() {
    local package="$1"
    log_info "Installing $package..."
    
    if dpkg -l | grep -q "^ii.*$package"; then
        log_info "$package is already installed, skipping."
        return 0
    fi
    
    if apt-get install -y --no-install-recommends "$package"; then
        log_success "$package installed successfully."
        return 0
    else
        log_warn "$package installation failed, trying with --fix-broken..."
        apt-get install -f -y
        if apt-get install -y --no-install-recommends "$package"; then
            log_success "$package installed successfully after recovery."
            return 0
        else
            log_error "$package installation failed."
            return 1
        fi
    fi
}

install_packages() {
    local category="$1"
    shift
    local packages=("$@")
    
    log_info "Installing $category tools..."
    
    for pkg in "${packages[@]}"; do
        install_package "$pkg"
    done
    
    log_info "$category tools installation completed."
}

install_security_tools() {
    local security_packages=(
        "fail2ban"
        "rkhunter"
        "chkrootkit"
        "lynis"
        "aide"
        "auditd"
        "libpam-pwquality"
        "acct"
        "sysstat"
        "apparmor"
        "apparmor-utils"
        "apparmor-profiles"
        "cryptsetup"
        "ufw"
        "clamav"
        "clamav-daemon"
        "clamtk"
    )
    
    install_packages "Security" "${security_packages[@]}"
}

install_monitoring_tools() {
    local monitoring_packages=(
        "iotop"
        "iftop"
        "nethogs"
        "nload"
        "tcpdump"
        "nmap"
        "netcat-openbsd"
        "lsof"
        "psmisc"
        "net-tools"
        "dstat"
        "atop"
        "htop"
        "glances"
        "vnstat"
        "bmon"
        "speedtest-cli"
        "traceroute"
        "whois"
    )
    
    install_packages "Monitoring" "${monitoring_packages[@]}"
}

install_system_utilities() {
    local system_packages=(
        "etckeeper"
        "mtr"
        "traceroute"
        "molly-guard"
        "ncdu"
        "duplicity"
        "rsync"
        "screen"
        "tmux"
        "unzip"
        "zip"
        "p7zip-full"
        "git"
        "git-lfs"
        "curl"
        "wget"
        "jq"
        "gnupg"
        "software-properties-common"
        "build-essential"
        "apt-transport-https"
        "ca-certificates"
        "nano"
        "vim"
        "tree"
    )
    
    install_packages "System Utilities" "${system_packages[@]}"
}

configure_fail2ban() {
    log_info "Configuring fail2ban..."
    
    # Only configure if fail2ban is installed
    if dpkg -l | grep -q "^ii.*fail2ban"; then
        # Create backup of original config if it exists
        if [ -f /etc/fail2ban/jail.conf ]; then
            cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.conf.backup.$(date +%Y%m%d)
        fi
        
        # Create local configuration
        cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
banaction = iptables-multiport

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 7200

[nginx-http-auth]
enabled = true
port = http,https
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[apache-auth]
enabled = true
port = http,https
filter = apache-auth
logpath = /var/log/apache*/*error.log
maxretry = 3
EOF
        
        # Restart and enable service
        systemctl enable fail2ban
        systemctl restart fail2ban
        log_success "fail2ban configured and enabled"
    else
        log_warn "fail2ban not installed, skipping configuration"
    fi
}

configure_aide() {
    log_info "Configuring AIDE (Advanced Intrusion Detection Environment)..."
    
    # Only configure if AIDE is installed
    if dpkg -l | grep -q "^ii.*aide"; then
        # Create AIDE configuration backup
        if [ -f /etc/aide/aide.conf ]; then
            cp /etc/aide/aide.conf /etc/aide/aide.conf.backup.$(date +%Y%m%d)
        fi
        
        # Initialize AIDE database
        log_info "Initializing AIDE database (this may take a while)..."
        aideinit -y || { 
            log_warn "AIDE initialization failed, continuing...";
            return;
        }
        
        # Only move the DB file if it was created
        if [ -f /var/lib/aide/aide.db.new ]; then
            cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db
            log_success "AIDE database initialized and configured."
            
            # Create a daily check script
            cat > /etc/cron.daily/aide-check << 'EOF'
#!/bin/bash
# Daily AIDE check
/usr/bin/aide.wrapper --check > /var/log/aide/aide-check-$(date +%Y%m%d).log
EOF
            chmod +x /etc/cron.daily/aide-check
            
            # Create log directory if it doesn't exist
            mkdir -p /var/log/aide
            log_success "AIDE daily check configured."
        else
            log_warn "AIDE database not created, skipping further configuration."
        fi
    else
        log_warn "AIDE not installed, skipping configuration."
    fi
}

configure_auditd() {
    log_info "Configuring auditd..."
    
    # Only configure if auditd is installed
    if dpkg -l | grep -q "^ii.*auditd"; then
        # Create rules directory if it doesn't exist
        mkdir -p /etc/audit/rules.d/
        
        # Backup existing rules if present
        if [ -f /etc/audit/rules.d/audit.rules ]; then
            cp /etc/audit/rules.d/audit.rules /etc/audit/rules.d/audit.rules.backup.$(date +%Y%m%d)
        fi
        
        # Create comprehensive auditing rules
        cat > /etc/audit/rules.d/audit.rules << 'EOF'
# Audit system calls
-a exit,always -F arch=b64 -S execve -k exec_commands
-a exit,always -F arch=b32 -S execve -k exec_commands

# Monitor file system mounts
-a always,exit -F arch=b64 -S mount -F auid>=1000 -F auid!=4294967295 -k mounts
-a always,exit -F arch=b32 -S mount -F auid>=1000 -F auid!=4294967295 -k mounts

# Monitor authentication modifications
-w /etc/passwd -p wa -k auth_changes
-w /etc/shadow -p wa -k auth_changes
-w /etc/group -p wa -k auth_changes
-w /etc/gshadow -p wa -k auth_changes
-w /etc/security/opasswd -p wa -k auth_changes

# Monitor network configuration changes
-w /etc/network/ -p wa -k network_changes
-w /etc/netplan/ -p wa -k network_changes
-w /etc/hosts -p wa -k network_changes
-w /etc/resolv.conf -p wa -k network_changes
-w /etc/sysconfig/network -p wa -k network_changes

# Monitor system configuration changes
-w /etc/security/ -p wa -k security_changes
-w /etc/sudoers -p wa -k sudo_changes
-w /etc/sudoers.d/ -p wa -k sudo_changes
-w /etc/ssh/sshd_config -p wa -k sshd_config

# Monitor package management
-w /usr/bin/dpkg -p x -k package_changes
-w /usr/bin/apt -p x -k package_changes
-w /usr/bin/apt-get -p x -k package_changes
-w /usr/bin/apt-add-repository -p x -k package_changes

# Monitor logs and configuration files
-w /var/log/ -p wa -k log_changes
-w /etc/ -p wa -k config_changes

# Monitor user and group management
-w /usr/sbin/useradd -p x -k user_modification
-w /usr/sbin/usermod -p x -k user_modification
-w /usr/sbin/userdel -p x -k user_modification
-w /usr/sbin/groupadd -p x -k group_modification
-w /usr/sbin/groupmod -p x -k group_modification
-w /usr/sbin/groupdel -p x -k group_modification

# Monitor privileged command execution
-a always,exit -F path=/usr/bin/sudo -F perm=x -F auid>=1000 -F auid!=4294967295 -k privileged_command
-a always,exit -F path=/bin/su -F perm=x -F auid>=1000 -F auid!=4294967295 -k privileged_command

# Set a reasonable size limit for the audit logs
-e 2
EOF
        
        # Restart auditd service
        systemctl restart auditd || log_warn "Failed to restart auditd, continuing..."
        log_success "auditd configured with comprehensive rules"
    else
        log_warn "auditd not installed, skipping configuration"
    fi
}

apply_system_hardening() {
    log_info "Applying additional system hardening..."
    
    # Apply /tmp hardening 
    if ! grep -q "/tmp" /etc/fstab; then
        log_info "Adding secure /tmp mount to fstab..."
        echo "tmpfs     /tmp     tmpfs     defaults,rw,nosuid,nodev,noexec,relatime,size=2G     0     0" >> /etc/fstab
    fi
    
    # Apply /run/shm hardening
    if ! grep -q "/run/shm.*noexec" /etc/fstab; then
        log_info "Adding secure /run/shm mount to fstab..."
        echo "tmpfs     /run/shm     tmpfs     defaults,noexec,nosuid     0     0" >> /etc/fstab
    fi
    
    # Configure core dumps
    if ! grep -q "* hard core 0" /etc/security/limits.conf; then
        log_info "Disabling core dumps..."
        echo "* hard core 0" >> /etc/security/limits.conf
        echo "fs.suid_dumpable = 0" >> /etc/sysctl.conf
    fi
    
    # Configure system-wide security parameters
    log_info "Setting security-related sysctl parameters..."
    
    # Create sysctl security configuration
    cat > /etc/sysctl.d/10-security-hardening.conf << 'EOF'
# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Block SYN attacks
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Disable IP forwarding
net.ipv4.ip_forward = 0

# Disable IPv6 if not needed
# Uncomment these lines if IPv6 is not required
# net.ipv6.conf.all.disable_ipv6 = 1
# net.ipv6.conf.default.disable_ipv6 = 1
# net.ipv6.conf.lo.disable_ipv6 = 1

# Protect against TCP time-wait assassination
net.ipv4.tcp_rfc1337 = 1

# Protect against bad error messages
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Restrict kernel pointer access
kernel.kptr_restrict = 2

# Restrict dmesg access
kernel.dmesg_restrict = 1

# Restrict perf usage for unprivileged users
kernel.perf_event_paranoid = 3

# Restrict ptrace scope
kernel.yama.ptrace_scope = 1

# Enable ASLR
kernel.randomize_va_space = 2

# Increase file descriptor limit
fs.file-max = 65535

# TCP BBR congestion control for better performance
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
EOF
    
    # Apply sysctl settings
    sysctl -p /etc/sysctl.d/10-security-hardening.conf || log_warn "Some sysctl settings might not have applied correctly"
    
    # Improve password security
    if [ -f /etc/pam.d/common-password ]; then
        log_info "Enhancing password policies..."
        # Backup original file
        cp /etc/pam.d/common-password /etc/pam.d/common-password.backup.$(date +%Y%m%d)
        
        # Check if pwquality is already configured
        if ! grep -q "pam_pwquality.so" /etc/pam.d/common-password; then
            # Add pwquality requirements to PAM
            sed -i '/pam_unix.so/i password requisite pam_pwquality.so retry=3 minlen=12 difok=3 ucredit=-1 lcredit=-1 dcredit=-1 ocredit=-1 reject_username enforce_for_root' /etc/pam.d/common-password
        fi
    fi
    
    # Setup grub password if it doesn't exist
    if [ -f /etc/grub.d/40_custom ] && ! grep -q "password" /etc/grub.d/40_custom; then
        log_info "Setting up GRUB password protection..."
        # Generate password hash
        GRUB_PASSWORD=$(openssl passwd -6 "SecureGrubPassword")
        # Add to grub configuration
        cat >> /etc/grub.d/40_custom << EOF
set superusers="admin"
password_pbkdf2 admin $GRUB_PASSWORD
EOF
        # Update grub configuration
        update-grub || log_warn "Failed to update GRUB, continuing..."
    fi
    
    log_success "System hardening applied"
}

configure_logging() {
    log_info "Configuring system logging..."
    
    # Create backup of rsyslog.conf if it exists
    if [ -f /etc/rsyslog.conf ]; then
        cp /etc/rsyslog.conf /etc/rsyslog.conf.backup.$(date +%Y%m%d)
    fi
    
    # Add enhanced logging configuration
    cat > /etc/rsyslog.d/90-enhanced-logging.conf << 'EOF'
# Enhanced logging configuration

# Log auth messages to a separate file
auth,authpriv.*                 /var/log/auth.log

# Log all kernel messages to the console
#kern.*                          /dev/console

# Log all kernel messages to a file
kern.*                          /var/log/kern.log

# Log cron jobs
cron.*                          /var/log/cron.log

# Log daemon messages
daemon.*                        /var/log/daemon.log

# Log mail messages
mail.*                          /var/log/mail.log

# Everybody gets emergency messages
*.emerg                         :omusrmsg:*

# Save boot messages also to boot.log
local7.*                        /var/log/boot.log

# Remote logging (uncomment and adjust for your remote log server)
# *.* @@remote-host:514
EOF
    
    # Create logrotate config if it doesn't exist
    if [ ! -f "/etc/logrotate.d/custom" ]; then
        cat > /etc/logrotate.d/custom << 'EOF'
/var/log/auth.log
/var/log/kern.log
/var/log/syslog
/var/log/cron.log
/var/log/daemon.log
/var/log/mail.log
/var/log/boot.log
{
    rotate 14
    daily
    compress
    delaycompress
    missingok
    notifempty
    create 0640 syslog adm
    sharedscripts
    postrotate
        /usr/lib/rsyslog/rsyslog-rotate
    endscript
}
EOF
    fi
    
    # Restart rsyslog service
    systemctl restart rsyslog || log_warn "Failed to restart rsyslog, continuing..."
    
    log_success "System logging configured"
}

configure_firewall() {
    log_info "Configuring UFW firewall..."
    
    # Check if UFW is installed
    if ! command -v ufw >/dev/null 2>&1; then
        log_warn "UFW not installed. Installing..."
        install_package "ufw"
    fi
    
    # Reset UFW to default
    ufw --force reset
    
    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow ssh
    
    # Allow common services - uncomment as needed
    # ufw allow 80/tcp      # HTTP
    # ufw allow 443/tcp     # HTTPS
    # ufw allow 22/tcp      # SSH (redundant with the above 'allow ssh' rule)
    # ufw allow 53          # DNS
    
    # Allow Samba if installed
    if dpkg -l | grep -q "^ii.*samba"; then
        log_info "Allowing Samba in firewall..."
        ufw allow samba
    fi
    
    # Enable IPv6 support
    sed -i 's/IPV6=no/IPV6=yes/' /etc/default/ufw
    
    # Enable firewall
    echo "y" | ufw enable
    
    # Show status
    ufw status verbose
    
    log_success "Firewall configured and enabled"
}

install_security_audit_tools() {
    log_info "Installing and configuring security audit tools..."
    
    # Install Lynis if not already installed
    if ! dpkg -l | grep -q "^ii.*lynis"; then
        install_package "lynis"
    fi
    
    # Create a script to run regular security audits
    cat > /usr/local/bin/security-audit.sh << 'EOF'
#!/bin/bash
# Security Audit Script

LOG_DIR="/var/log/security-audits"
DATE=$(date +%Y%m%d)
HOSTNAME=$(hostname)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run various security checks

# Lynis audit
if command -v lynis >/dev/null 2>&1; then
    echo "Running Lynis audit..."
    lynis audit system --quiet > "$LOG_DIR/lynis-audit-$HOSTNAME-$DATE.log"
fi

# RKHunter check
if command -v rkhunter >/dev/null 2>&1; then
    echo "Running RKHunter check..."
    rkhunter --check --skip-keypress > "$LOG_DIR/rkhunter-check-$HOSTNAME-$DATE.log"
fi

# ChkRootKit check
if command -v chkrootkit >/dev/null 2>&1; then
    echo "Running ChkRootKit check..."
    chkrootkit > "$LOG_DIR/chkrootkit-$HOSTNAME-$DATE.log"
fi

# AIDE check if database exists
if command -v aide >/dev/null 2>&1 && [ -f /var/lib/aide/aide.db ]; then
    echo "Running AIDE check..."
    aide --check > "$LOG_DIR/aide-check-$HOSTNAME-$DATE.log"
fi

# ClamAV scan
if command -v clamscan >/dev/null 2>&1; then
    echo "Running ClamAV scan on /home directories..."
    clamscan --recursive --infected /home > "$LOG_DIR/clamscan-home-$HOSTNAME-$DATE.log"
fi

# Check users with empty passwords
echo "Checking for users with empty passwords..."
awk -F: '($2 == "" || $2 == "!") {print $1}' /etc/shadow > "$LOG_DIR/empty-passwords-$HOSTNAME-$DATE.log"

# Check for users with UID 0 (other than root)
echo "Checking for users with UID 0..."
awk -F: '($3 == 0 && $1 != "root") {print $1}' /etc/passwd > "$LOG_DIR/uid0-users-$HOSTNAME-$DATE.log"

# Check for world-writable files
echo "Checking for world-writable files (this may take a while)..."
find / -xdev -type f -perm -0002 -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null > "$LOG_DIR/world-writable-files-$HOSTNAME-$DATE.log"

# Check for world-writable directories
echo "Checking for world-writable directories..."
find / -xdev -type d -perm -0002 -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null > "$LOG_DIR/world-writable-dirs-$HOSTNAME-$DATE.log"

# Check for unowned files
echo "Checking for unowned files and directories..."
find / -xdev \( -nouser -o -nogroup \) -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null > "$LOG_DIR/unowned-files-$HOSTNAME-$DATE.log"

# Check for setuid files
echo "Checking for setuid files..."
find / -xdev -type f -perm -4000 -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null > "$LOG_DIR/setuid-files-$HOSTNAME-$DATE.log"

# Check failed logins
echo "Checking for failed login attempts..."
grep "Failed password" /var/log/auth.log | tail -n 1000 > "$LOG_DIR/failed-logins-$HOSTNAME-$DATE.log"

# Report generation
echo "Security audit completed on $(date)" > "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "------------------------------------------------------" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "Empty password accounts: $(wc -l < "$LOG_DIR/empty-passwords-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "UID 0 accounts (non-root): $(wc -l < "$LOG_DIR/uid0-users-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "World-writable files: $(wc -l < "$LOG_DIR/world-writable-files-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "World-writable directories: $(wc -l < "$LOG_DIR/world-writable-dirs-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "Unowned files and directories: $(wc -l < "$LOG_DIR/unowned-files-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "SetUID files: $(wc -l < "$LOG_DIR/setuid-files-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "Failed login attempts: $(wc -l < "$LOG_DIR/failed-logins-$HOSTNAME-$DATE.log")" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "------------------------------------------------------" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"
echo "Full logs available in $LOG_DIR" >> "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt"

# Email summary if mail is configured (uncomment and adjust as needed)
# cat "$LOG_DIR/audit-summary-$HOSTNAME-$DATE.txt" | mail -s "Security Audit Summary for $HOSTNAME on $DATE" root@localhost

echo "Security audit completed. Logs saved to $LOG_DIR/"
EOF
    
    # Make it executable
    chmod +x /usr/local/bin/security-audit.sh
    
    # Add a weekly cron job for this script
    cat > /etc/cron.weekly/security-audit << 'EOF'
#!/bin/bash
/usr/local/bin/security-audit.sh
EOF
    
    chmod +x /etc/cron.weekly/security-audit
    
    log_success "Security audit tools installed and configured for weekly runs"
}

configure_ssh() {
    log_info "Configuring SSH hardening..."
    
    # Check if SSH is installed
    if ! dpkg -l | grep -q "^ii.*openssh-server"; then
        log_warn "OpenSSH server not installed. Installing..."
        install_package "openssh-server"
    fi
    
    # Backup original config
    if [ -f /etc/ssh/sshd_config ]; then
        cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup.$(date +%Y%m%d)
    fi
    
    # Create hardened SSH config
    cat > /etc/ssh/sshd_config << 'EOF'
# SSH Server Configuration
# Hardened for security

Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key

# Logging
SyslogFacility AUTH
LogLevel VERBOSE

# Authentication
LoginGraceTime 30
PermitRootLogin no
StrictModes yes
MaxAuthTries 3
MaxSessions 5

# Public key authentication
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys

# Password authentication
PasswordAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no

# Kerberos options
KerberosAuthentication no
GSSAPIAuthentication no

# X11 forwarding
X11Forwarding no

# Security
UsePrivilegeSeparation sandbox
PermitUserEnvironment no
AllowAgentForwarding no
AllowTcpForwarding no
GatewayPorts no
X11UseLocalhost yes
PermitTunnel no
ClientAliveInterval 300
ClientAliveCountMax 3
UsePAM yes
UseDNS no

# Accepted environment variables
AcceptEnv LANG LC_*

# Subsystem
Subsystem sftp /usr/lib/openssh/sftp-server -f AUTHPRIV -l INFO

# Allow only specific users (uncomment and adjust as needed)
# AllowUsers username1 username2
EOF
    
    # Test the configuration
    sshd -t
    if [ $? -eq 0 ]; then
        log_success "SSH configuration is valid"
        # Restart SSH service
        systemctl restart ssh || systemctl restart sshd
    else
        log_error "SSH configuration has errors, reverting to backup"
        # Restore backup if the new config is invalid
        cp /etc/ssh/sshd_config.backup.$(date +%Y%m%d) /etc/ssh/sshd_config
        systemctl restart ssh || systemctl restart sshd
    fi
}

cleanup() {
    log_info "Performing final cleanup..."
    
    # Clean package cache
    apt-get autoremove -y
    apt-get clean
    
    # Final system updates
    apt-get update
    apt-get upgrade -y
    
    # Remove temporary files
    find /tmp -type f -atime +7 -delete
    
    log_success "Cleanup completed"
}

main() {
    check_root
    update_system
    
    # Install tools by category
    install_security_tools
    install_monitoring_tools
    install_system_utilities
    
    # Configure security features
    configure_fail2ban
    configure_aide
    configure_auditd
    apply_system_hardening
    configure_logging
    configure_firewall
    install_security_audit_tools
    configure_ssh
    
    # Final cleanup
    cleanup
    
    # Determine if there were any failures
    if [ -s "$FAILED_TOOLS_LOG" ]; then
        FAILED_COUNT=$(wc -l < "$FAILED_TOOLS_LOG")
        log_warn "Installation completed with $FAILED_COUNT errors. See $FAILED_TOOLS_LOG for details."
    else
        log_success "All installations completed successfully!"
    fi
    
    log_info "
✨ Security and System Tools Setup Complete! ✨

System is now enhanced with:
- Comprehensive security tools (fail2ban, auditd, AIDE, etc.)
- Advanced system monitoring (iotop, iftop, nethogs, etc.)
- System utilities and essential tools
- Hardened system configuration
- Enhanced logging setup
- Automated security audits (weekly)
- Firewall protection

Next steps:
1. Reboot the system to apply all changes
2. Review the first security audit report
3. Set up monitoring alert notifications
4. Check system logs for any issues

📝 Full installation log: $LOG_FILE
"

    # Check if any tools failed to install
    if [ -s "$FAILED_TOOLS_LOG" ]; then
        log_warn "The following tools had installation issues:
$(cat "$FAILED_TOOLS_LOG" | sed 's/^/- /')

You may want to try installing them manually."
    fi
}

# Execute main function
main
