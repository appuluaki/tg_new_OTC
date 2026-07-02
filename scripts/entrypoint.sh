#!/bin/sh
# entrypoint.sh — load /app/.env if present and export variables, then exec the CMD

for ENV_FILE in /app/config/.env /app/.env; do
  if [ -f "$ENV_FILE" ]; then
    # Export variables defined in .env (ignore comments and empty lines)
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  fi
done

exec "$@"
