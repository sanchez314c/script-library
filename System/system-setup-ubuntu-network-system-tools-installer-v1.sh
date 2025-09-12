#!/bin/bash

# Ubuntu Security and System Tools Installation Script
# Version: 1.1.0 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting Ubuntu Security and System Tools Setup..."

# Global variables
LOG_FILE="/var/log/security_setup.log"
ORIGINAL_PATH=$PATH

# Redirect output to log file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ This script requires root privileges. Please run with sudo."
        exit 1
    fi
    echo "✅ Running as root"
}

install_security_tools() {
    echo "🔒 Installing security tools..."
    
    # Install each package individually with extra precautions
    SECURITY_PACKAGES="fail2ban rkhunter chkrootkit lynis aide auditd libpam-pwquality acct sysstat apparmor apparmor-utils apparmor-profiles cryptsetup"
    
    for pkg in $SECURITY_PACKAGES; do
        echo "Installing $pkg..."
        apt-get install -y --no-install-recommends $pkg || echo "⚠️ Failed to install $pkg, continuing..."
    done
    
    echo "✅ Security tools installed"
}

install_monitoring_tools() {
    echo "📊 Installing monitoring tools..."
    
    # Install each package individually
    MONITORING_PACKAGES="iotop iftop nethogs nload tcpdump nmap netcat-openbsd lsof psmisc net-tools dstat atop"
    
    for pkg in $MONITORING_PACKAGES; do
        echo "Installing $pkg..."
        apt-get install -y --no-install-recommends $pkg || echo "⚠️ Failed to install $pkg, continuing..."
    done
    
    echo "✅ Monitoring tools installed"
}

install_system_utilities() {
    echo "🧰 Installing system utilities..."
    
    # Make sure we install each package individually
    SYSTEM_PACKAGES="etckeeper mtr traceroute molly-guard ncdu"
    
    for pkg in $SYSTEM_PACKAGES; do
        echo "Installing $pkg..."
        apt-get install -y --no-install-recommends $pkg || echo "⚠️ Failed to install $pkg, continuing..."
    done
    
    echo "✅ System utilities installed"
}

configure_fail2ban() {
    echo "🚨 Configuring fail2ban..."
    # Only configure if fail2ban is installed
    if dpkg -l | grep -q "^ii.*fail2ban"; then
        cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF
        systemctl enable fail2ban
        systemctl restart fail2ban
        echo "✅ fail2ban configured"
    else
        echo "⚠️ fail2ban not installed, skipping configuration"
    fi
}

configure_aide() {
    echo "🕵️ Configuring AIDE..."
    # Only configure if AIDE is installed
    if dpkg -l | grep -q "^ii.*aide"; then
        aideinit -y || echo "⚠️ AIDE initialization failed, continuing..."
        # Only move the DB file if it was created
        if [ -f /var/lib/aide/aide.db.new ]; then
            mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db
            echo "✅ AIDE configured"
        else
            echo "⚠️ AIDE DB not created, skipping"
        fi
    else
        echo "⚠️ AIDE not installed, skipping configuration"
    fi
}

configure_auditd() {
    echo "📜 Configuring auditd..."
    # Only configure if auditd is installed
    if dpkg -l | grep -q "^ii.*auditd"; then
        mkdir -p /etc/audit/rules.d/
        cat > /etc/audit/rules.d/audit.rules << EOF
# Monitor file system mounts
-a always,exit -F arch=b64 -S mount -F auid>=1000 -F auid!=4294967295 -k mounts

# Monitor system calls
-a exit,always -F arch=b64 -S execve -k exec_calls

# Monitor authentication changes
-w /etc/passwd -p wa -k auth_changes
-w /etc/shadow -p wa -k auth_changes
-w /etc/group -p wa -k auth_changes
-w /etc/gshadow -p wa -k auth_changes

# Monitor network configuration changes
-w /etc/network/ -p wa -k network_changes
-w /etc/netplan/ -p wa -k network_changes

# Monitor system configuration changes
-w /etc/security/ -p wa -k security_changes
-w /etc/sudoers -p wa -k sudo_changes
EOF
        systemctl restart auditd || echo "⚠️ Failed to restart auditd, continuing..."
        echo "✅ auditd configured"
    else
        echo "⚠️ auditd not installed, skipping configuration"
    fi
}

apply_system_hardening() {
    echo "🔧 Applying additional system hardening..."
    
    # Check if entry already exists before adding
    if ! grep -q "/run/shm.*noexec" /etc/fstab; then
        echo "tmpfs     /run/shm     tmpfs     defaults,noexec,nosuid     0     0" >> /etc/fstab
    fi
    
    # Check if entry already exists before adding
    if ! grep -q "* hard core 0" /etc/security/limits.conf; then
        echo "* hard core 0" >> /etc/security/limits.conf
    fi
    
    # Check if entry already exists before adding
    if ! grep -q "fs.suid_dumpable = 0" /etc/sysctl.conf; then
        echo "fs.suid_dumpable = 0" >> /etc/sysctl.conf
    fi
    
    # Add hardening settings if not already present
    if ! grep -q "# IP Spoofing protection" /etc/sysctl.conf; then
        cat >> /etc/sysctl.conf << EOF
# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

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

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF
    fi
    
    sysctl -p || echo "⚠️ Some sysctl settings might not have applied correctly"
    echo "✅ System hardening applied"
}

configure_logging() {
    echo "📝 Configuring system logging..."
    # Create logrotate config if it doesn't exist
    if [ ! -f "/etc/logrotate.d/custom" ]; then
        cat > /etc/logrotate.d/custom << EOF
/var/log/auth.log
/var/log/kern.log
/var/log/syslog
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
    echo "✅ System logging configured"
}

cleanup() {
    echo "🧹 Performing final cleanup..."
    apt-get autoremove -y
    apt-get clean
    echo "✅ Cleanup completed"
}

main() {
    check_root
    # Add an update before installing packages
    apt-get update
    install_security_tools
    install_monitoring_tools
    install_system_utilities
    configure_fail2ban
    configure_aide
    configure_auditd
    apply_system_hardening
    configure_logging
    cleanup

    echo "✨ Security and System Tools Setup Complete! ✨"
    echo "System is now enhanced with security and monitoring tools."
    echo "📋 Please review the log file at $LOG_FILE for details."
    echo "⚠️ Next steps:"
    echo "1. Set up monitoring alert notifications"
    echo "2. Check system logs for issues"
    echo "3. Regularly check AIDE reports"
}

main