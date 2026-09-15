#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLBOX_NAME="${AGENTARI_TOOLBOX_NAME:-agentari-dev}"

# On an immutable Linux host, keep package installation and compilation inside
# the project's development Toolbox.  The environment variable prevents the
# delegated invocation from entering the Toolbox recursively.
if [[ -z "${AGENTARI_IN_TOOLBOX:-}" ]] &&
   command -v toolbox >/dev/null 2>&1 &&
   [[ ! -e /.containerenv && ! -e /run/.containerenv ]]; then
    if ! toolbox run --container "$TOOLBOX_NAME" true >/dev/null 2>&1; then
        toolbox create "$TOOLBOX_NAME"
    fi
    exec env AGENTARI_IN_TOOLBOX=1 \
        toolbox run --container "$TOOLBOX_NAME" \
        "$PROJECT_ROOT/tools/install_and_run.sh" "$@"
fi

BUILD_DIR="${AGENTARI_BUILD_DIR:-$PROJECT_ROOT/build-agentari}"
BUILD_TYPE="${AGENTARI_BUILD_TYPE:-Release}"
BUILD_JOBS="${AGENTARI_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}"
BUILD_APP=OFF
ENABLE_VULKAN="${AGENTARI_ENABLE_VULKAN:-ON}"
RUN_TARGET=agentari-console
NO_RUN=0
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage: tools/install_and_run.sh [--app|--console] [--no-run] [options]

Installs build dependencies, configures AgentAri, builds it, runs the tests,
and launches the console. All remaining options are passed to the executable.

  --console       Build and run the SSH-friendly console (default)
  --app           Build and run the SDL application
  --no-run        Install, build, and test without launching the executable
  --help          Show this help

Examples:
  tools/install_and_run.sh
  tools/install_and_run.sh --console --resume-state --state agentari-state.txt
  tools/install_and_run.sh --app --vulkan
  tools/install_and_run.sh --no-run

Environment:
  AGENTARI_BUILD_DIR, AGENTARI_BUILD_TYPE, AGENTARI_BUILD_JOBS
  AGENTARI_ENABLE_VULKAN, AGENTARI_TOOLBOX_NAME
EOF
}

while (($# > 0)); do
    case "$1" in
    --console)
        BUILD_APP=OFF
        RUN_TARGET=agentari-console
        shift
        ;;
    --app)
        BUILD_APP=ON
        RUN_TARGET=agent-ari
        shift
        ;;
    --no-run)
        NO_RUN=1
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    --)
        shift
        RUN_ARGS+=("$@")
        break
        ;;
    *)
        RUN_ARGS+=("$1")
        shift
        ;;
    esac
done

run_privileged() {
    if ((EUID == 0)); then
        "$@"
    else
        if ! command -v sudo >/dev/null 2>&1; then
            printf '%s\n' 'This installation needs sudo, but sudo was not found.' >&2
            exit 1
        fi
        sudo "$@"
    fi
}

install_packages() {
    local package_manager

    if command -v dnf >/dev/null 2>&1; then
        package_manager=dnf
        run_privileged "$package_manager" install -y \
            gcc-c++ cmake ninja-build pkgconf-pkg-config python3 git
        if [[ "$BUILD_APP" == ON ]]; then
            run_privileged "$package_manager" install -y SDL3-devel
        fi
        if [[ "$ENABLE_VULKAN" == ON ]]; then
            for package in vulkan-headers vulkan-loader-devel glslc; do
                if ! run_privileged "$package_manager" install -y "$package"; then
                    printf 'Optional Vulkan package unavailable: %s\n' "$package" >&2
                fi
            done
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        package_manager=apt-get
        run_privileged "$package_manager" update
        run_privileged "$package_manager" install -y \
            build-essential cmake ninja-build pkg-config python3 git
        if [[ "$BUILD_APP" == ON ]]; then
            run_privileged "$package_manager" install -y libsdl3-dev
        fi
        if [[ "$ENABLE_VULKAN" == ON ]]; then
            for package in libvulkan-dev glslc; do
                if ! run_privileged "$package_manager" install -y "$package"; then
                    printf 'Optional Vulkan package unavailable: %s\n' "$package" >&2
                fi
            done
        fi
    elif command -v pacman >/dev/null 2>&1; then
        package_manager=pacman
        run_privileged "$package_manager" -Sy --needed --noconfirm \
            base-devel cmake ninja pkgconf python git
        if [[ "$BUILD_APP" == ON ]]; then
            run_privileged "$package_manager" -Sy --needed --noconfirm sdl3
        fi
        if [[ "$ENABLE_VULKAN" == ON ]]; then
            run_privileged "$package_manager" -Sy --needed --noconfirm \
                vulkan-headers vulkan-icd-loader glslc ||
                printf '%s\n' 'Optional Vulkan packages unavailable.' >&2
        fi
    else
        printf '%s\n' 'Unsupported distribution: install C++20, CMake, Ninja, Git, and Python 3 manually.' >&2
        exit 1
    fi
}

install_packages

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DAGENTARI_BUILD_APP="$BUILD_APP" \
    -DAGENTARI_ENABLE_VULKAN="$ENABLE_VULKAN"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$BUILD_DIR" --output-on-failure

if ((NO_RUN)); then
    printf 'AgentAri is built and tested in %s\n' "$BUILD_DIR"
    exit 0
fi

cd "$PROJECT_ROOT"
exec "$BUILD_DIR/$RUN_TARGET" "${RUN_ARGS[@]}"
