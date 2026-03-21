#!/bin/bash
# VSCode integrated terminal init — auto-loaded for every terminal in this workspace.
# Sources the user's normal bashrc, then loads the project module and venv.

# Load user's normal shell config first
[ -f ~/.bashrc ] && source ~/.bashrc

pick_codeserver_node() {
    command -v squeue >/dev/null 2>&1 || return 1
    squeue -h -u "$USER" -n bc_alvis_codeserver -o "%N %L" 2>/dev/null | awk '
        function tosec(t, a, n, d, h, m, s) {
            if (t == "" || t == "N/A" || t == "UNLIMITED") return -1
            d = 0
            if (index(t, "-")) {
                split(t, a, "-")
                d = a[1]
                t = a[2]
            }
            n = split(t, a, ":")
            if (n == 3) {
                h = a[1]; m = a[2]; s = a[3]
            } else if (n == 2) {
                h = 0; m = a[1]; s = a[2]
            } else {
                h = 0; m = 0; s = a[1]
            }
            return d * 86400 + h * 3600 + m * 60 + s
        }
        {
            node = $1
            sub(/,.*/, "", node)
            score = tosec($2)
            if (score > best) {
                best = score
                bestnode = node
            }
        }
        END {
            if (bestnode != "") print bestnode
        }'
}

NODE="${DITUNET_NODE:-$(pick_codeserver_node)}"
NODE="${NODE:-alvis9-05}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Jump to compute node for integrated terminals.
# The previous command only tested SSH and returned immediately.
if [ -z "${DITUNET_ON_NODE:-}" ] && [ "$(hostname -s)" != "$NODE" ]; then
    exec ssh -tt -o ConnectTimeout=5 "$NODE" \
        "export DITUNET_ON_NODE=1; cd \"$PROJECT_DIR\"; \
         source /usr/share/lmod/lmod/init/bash; \
         [ -f \"$PROJECT_DIR/.ditunet_venv/bin/activate\" ]; \
         exec bash -i"
fi

# Initialize lmod if not already available
if ! command -v module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash
fi

# Load the Python module locally if not on target node
module load Python/3.11.3-GCCcore-12.3.0

# Activate the project venv
VENV="$PROJECT_DIR/.ditunet_venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
