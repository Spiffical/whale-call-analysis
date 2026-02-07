#!/usr/bin/env bash
set -euo pipefail

# Mount ONC whalestor CIFS share.
# Defaults match the working mount settings discussed in this repo.

SERVER="${WHALESTOR_SERVER:-142.104.198.122}"
SHARE="${WHALESTOR_SHARE:-Hydrophone}"
SUBDIR="${WHALESTOR_SUBDIR:-HydrophoneData}"
MOUNT_POINT="${WHALESTOR_MOUNT_POINT:-$HOME/whalestor_mount}"
USERNAME="${WHALESTOR_USERNAME:-$USER}"
DOMAIN="${WHALESTOR_DOMAIN:-onc}"
CREDENTIALS_FILE="${WHALESTOR_CREDENTIALS_FILE:-}"
SKIP_NETWORK_CHECK=0

usage() {
  cat <<'EOF'
Usage: mount_whalestor.sh [options]

Options:
  --server <host-or-ip>         CIFS host (default: 142.104.198.122)
  --share <share-name>          CIFS share name (default: Hydrophone)
  --subdir <path>               Informational subdir under mount (default: HydrophoneData)
  --mount-point <path>          Local mount point (default: ~/whalestor_mount)
  --username <name>             CIFS username (default: current user)
  --domain <name>               CIFS domain (default: onc)
  --credentials-file <path>     Optional cifs credentials file
  --skip-network-check          Skip SMB port preflight check
  -h, --help                    Show this help

Credentials file format (chmod 600):
  username=sbialek
  password=YOUR_PASSWORD
  domain=onc
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --share) SHARE="$2"; shift 2 ;;
    --subdir) SUBDIR="$2"; shift 2 ;;
    --mount-point) MOUNT_POINT="$2"; shift 2 ;;
    --username) USERNAME="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    --credentials-file) CREDENTIALS_FILE="$2"; shift 2 ;;
    --skip-network-check) SKIP_NETWORK_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v mount.cifs >/dev/null 2>&1; then
  echo "mount.cifs not found. Install with: sudo apt install -y cifs-utils" >&2
  exit 1
fi

if [[ "$SKIP_NETWORK_CHECK" -eq 0 ]]; then
  if command -v nc >/dev/null 2>&1; then
    if ! nc -vz -w 3 -4 "$SERVER" 445 >/dev/null 2>&1; then
      echo "Cannot reach $SERVER:445 from this VM." >&2
      echo "Check VPN/routing/firewall before mounting." >&2
      exit 1
    fi
  else
    echo "Warning: 'nc' not found, skipping SMB network preflight check." >&2
  fi
fi

mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
  echo "Already mounted: $MOUNT_POINT"
else
  OPTS=(
    "uid=$(id -u)"
    "gid=$(id -g)"
    "vers=3.0"
    "sec=ntlmssp"
    "file_mode=0664"
    "dir_mode=0775"
  )

  if [[ -n "$CREDENTIALS_FILE" ]]; then
    if [[ ! -f "$CREDENTIALS_FILE" ]]; then
      echo "Credentials file not found: $CREDENTIALS_FILE" >&2
      exit 1
    fi
    OPTS+=("credentials=$CREDENTIALS_FILE")
  else
    # Without credentials file, mount prompts for password.
    OPTS+=("username=$USERNAME" "domain=$DOMAIN")
  fi

  echo "Mounting //$SERVER/$SHARE -> $MOUNT_POINT"
  sudo mount -t cifs "//$SERVER/$SHARE" "$MOUNT_POINT" -o "$(IFS=,; echo "${OPTS[*]}")"
fi

echo "Mounted path: $MOUNT_POINT"
if [[ -n "$SUBDIR" ]]; then
  if [[ -d "$MOUNT_POINT/$SUBDIR" ]]; then
    echo "Data subdir: $MOUNT_POINT/$SUBDIR"
  else
    echo "Warning: subdir not found: $MOUNT_POINT/$SUBDIR" >&2
  fi
fi
