#!/usr/bin/env bash
# Run ON the EC2 box (Amazon Linux 2023), as ec2-user:  sudo bash ec2-setup.sh carlos-rag.duckdns.org
# Installs the Streamlit frontend as a locked-down service behind Caddy (HTTPS). Safe to re-run.
set -euo pipefail
DOMAIN="${1:?usage: sudo bash ec2-setup.sh <your-domain>}"
REPO=https://github.com/carloshernandez201/GeminiRAG-Embedding-OrchestrationE2E.git
APP=/opt/rag-app
CADDY_VERSION=2.10.0

echo "== Packages + automatic security updates"
dnf -y -q install git python3.12 dnf-automatic
sed -i 's/^upgrade_type.*/upgrade_type = security/; s/^apply_updates.*/apply_updates = yes/' /etc/dnf/automatic.conf
systemctl enable --now dnf-automatic.timer

echo "== App user + code"
id rag >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/rag --shell /sbin/nologin rag
if [ -d "$APP/.git" ]; then git -C "$APP" pull -q; else git clone -q "$REPO" "$APP"; fi
python3.12 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q -r "$APP/src/requirements.txt"
chown -R root:root "$APP"   # app code is read-only to the service user

echo "== Secrets (typed, not echoed, never in shell history)"
if [ ! -f /etc/rag-app.env ]; then
  read -rsp "INNGEST_EVENT_KEY: " EK; echo
  read -rsp "INNGEST_SIGNING_KEY: " SK; echo
  read -rp  "S3_BUCKET: " BUCKET
  install -m 600 -o rag -g rag /dev/null /etc/rag-app.env
  cat > /etc/rag-app.env <<ENV
INNGEST_EVENT_KEY=$EK
INNGEST_SIGNING_KEY=$SK
INNGEST_API_BASE=https://api.inngest.com/v1
S3_BUCKET=$BUCKET
AWS_DEFAULT_REGION=us-east-1
ENV
fi

echo "== Streamlit service (localhost only, sandboxed)"
pkill -f "streamlit run" || true
cat > /etc/systemd/system/rag-streamlit.service <<UNIT
[Unit]
Description=RAG Streamlit frontend
After=network-online.target

[Service]
User=rag
WorkingDirectory=$APP
EnvironmentFile=/etc/rag-app.env
ExecStart=$APP/venv/bin/streamlit run src/streamlit_app.py
Restart=always
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
ReadWritePaths=/var/lib/rag
Environment=HOME=/var/lib/rag

[Install]
WantedBy=multi-user.target
UNIT

echo "== Caddy $CADDY_VERSION (checksum-verified) for HTTPS"
if ! /usr/local/bin/caddy version 2>/dev/null | grep -q "v$CADDY_VERSION"; then
  ARCH=$(uname -m); [ "$ARCH" = "x86_64" ] && ARCH=amd64; [ "$ARCH" = "aarch64" ] && ARCH=arm64
  TMP=$(mktemp -d); cd "$TMP"
  TAR="caddy_${CADDY_VERSION}_linux_${ARCH}.tar.gz"
  curl -fsSLO "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/$TAR"
  curl -fsSLO "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_checksums.txt"
  grep " $TAR\$" "caddy_${CADDY_VERSION}_checksums.txt" | sha512sum -c -
  tar -xzf "$TAR" caddy && install -m 755 caddy /usr/local/bin/caddy
  cd / && rm -rf "$TMP"
fi
id caddy >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/caddy --shell /sbin/nologin caddy
mkdir -p /etc/caddy
cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
	encode gzip
	header {
		Strict-Transport-Security "max-age=31536000; includeSubDomains"
		X-Content-Type-Options nosniff
		X-Frame-Options DENY
		Referrer-Policy no-referrer
		-Server
	}
	request_body {
		max_size 12MB
	}
	reverse_proxy 127.0.0.1:8501
}
CADDY
cat > /etc/systemd/system/caddy.service <<UNIT
[Unit]
Description=Caddy
After=network-online.target

[Service]
User=caddy
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile
AmbientCapabilities=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/caddy
Environment=HOME=/var/lib/caddy
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now rag-streamlit caddy
systemctl restart rag-streamlit caddy
sleep 3
systemctl --no-pager --lines=0 status rag-streamlit caddy | grep -E "●|Active"
echo "Done. Open https://$DOMAIN (the first load can take ~30s while Caddy gets a certificate)."
