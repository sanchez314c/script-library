#!/bin/bash

# --- Configuration ---
PROJECT_DIR="open-webui-https"
CERT_DIR="certs"
NGINX_CONF_DIR="nginx_conf"
OPENWEBUI_DATA_DIR="open-webui-data"

# --- Helper function for robust command execution ---
run_command() {
    echo "Running: $*"
    if ! eval "$@"; then
        echo "ERROR: Command failed: $*"
        exit 1
    fi
}

# --- Check for Docker and stop conflicting containers ---
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in your PATH. Please install Docker Desktop and try again."
    exit 1
fi

echo "Stopping any potentially conflicting containers first..."
docker stop open-webui-nginx >/dev/null 2>&1 || true
docker rm open-webui-nginx >/dev/null 2>&1 || true
docker stop open-webui >/dev/null 2>&1 || true
docker rm open-webui >/dev/null 2>&1 || true
echo "Cleanup of old containers complete."


# --- Create Project Directory ---
echo "Creating project directory: ~/$PROJECT_DIR"
run_command "mkdir -p ~/$PROJECT_DIR"
run_command "cd ~/$PROJECT_DIR"


# --- Create Certificate Directory ---
echo "Creating certificates directory: $CERT_DIR"
run_command "mkdir -p \"$CERT_DIR\""

CERT_FULL_PATH="$PWD/$CERT_DIR/localhost.crt"
KEY_FULL_PATH="$PWD/$CERT_DIR/localhost.key"


# --- Attempt to delete existing localhost certificate from System Keychain ---
echo "Attempting to delete any old 'localhost' certificates from System Keychain..."
if security find-certificate -c "localhost" -k "/Library/Keychains/System.keychain" &>/dev/null; then
    echo "Found existing 'localhost' certificate. Deleting..."
    run_command "sudo security delete-certificate -c 'localhost' -k \"/Library/Keychains/System.keychain\""
else
    echo "No existing 'localhost' certificate found."
fi


# --- Generate Self-Signed SSL Certificate ---
echo "Generating self-signed SSL certificate for localhost..."
run_command "openssl req -x509 -newkey rsa:4096 -nodes -keyout \"$KEY_FULL_PATH\" -out \"$CERT_FULL_PATH\" -days 365 -subj \"/CN=localhost\" 2>/dev/null"


# --- Add the newly generated certificate to the System Keychain ---
echo "Adding new certificate to macOS System Keychain (requires password)..."
run_command "sudo security add-trusted-cert -d -r trustRoot -k \"/Library/Keychains/System.keychain\" \"$CERT_FULL_PATH\""
echo "Certificate added. Manual trust may still be required in Keychain Access."


# --- Create Nginx Configuration Directory and File ---
echo "Creating Nginx configuration..."
run_command "mkdir -p \"$NGINX_CONF_DIR\""
cat <<EOF > "$NGINX_CONF_DIR/default.conf"
server {
    listen 80;
    server_name localhost;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name localhost;

    ssl_certificate /etc/nginx/certs/localhost.crt;
    ssl_certificate_key /etc/nginx/certs/localhost.key;

    location / {
        proxy_pass http://open-webui:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF


# --- Create Docker Compose File ---
echo "Creating docker-compose.yml..."
cat <<EOF > docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:latest
    container_name: open-webui-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./$CERT_DIR:/etc/nginx/certs:ro
      - ./$NGINX_CONF_DIR/default.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - open-webui
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    volumes:
      - $OPENWEBUI_DATA_DIR:/app/backend/data
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  open-webui-data:

EOF

# --- Clean and Start Docker Compose ---
echo "Tearing down any old services..."
run_command "docker compose down --volumes"
echo "Starting Open WebUI and Nginx..."
run_command "docker compose up -d"

echo ""
echo "--- SETUP COMPLETE ---"
echo ""
echo "ACTION REQUIRED: Please perform the following steps:"
echo "1. Open 'Keychain Access' on your Mac."
echo "2. Select the 'System' keychain."
echo "3. Find the 'localhost' certificate and double-click it."
echo "4. Expand 'Trust' and set 'When using this certificate:' to 'Always Trust'."
echo "5. Close the window (enter password if prompted)."
echo "6. **QUIT AND RESTART YOUR WEB BROWSER.**"
echo ""
echo "Then, navigate to: https://localhost"
echo "After logging in, go to Settings -> Connections to add your llama.cpp server."