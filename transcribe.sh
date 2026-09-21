#!/usr/bin/env bash
set -euo pipefail

options=(--context-size 512)
uv_options=()
args=()
for arg in "$@"; do
    case "$arg" in
        --quality)
            options=(--model ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx --context-size 1024 --delay-ms 2400)
            ;;
        --translate-en)
            uv_options=(--extra translation)
            args+=("$arg")
            ;;
        --help|-h)
            printf '%s\n' \
                'Usage: ./transcribe.sh [--quality] [--translate-en] [voxmlx options]' \
                'Default: 6-bit model, 512-token context, 480 ms delay.' \
                'Quality: 8-bit model, 1024-token context, 2400 ms delay (M5 Max / 48 GB).' \
                '--translate-en: local French-to-English translation, displaying and saving both.' \
                'Translation dependencies and model are downloaded on first use.' \
                'Both modes show text live and save it in transcripts/. Ctrl+C stops.' \
                'Override settings with --model, --context-size, or --delay-ms.'
            exit 0
            ;;
        *) args+=("$arg") ;;
    esac
done

cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1
if [[ ${#uv_options[@]} -gt 0 ]]; then
    export XDG_DATA_HOME="${XDG_DATA_HOME:-$PWD/.cache/data}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
    export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$PWD/.cache/config}"
fi
mkdir -p transcripts
transcript="$PWD/transcripts/room-$(date +%Y%m%d-%H%M%S).txt"
printf 'Saving live transcript to: %s\nPress Ctrl+C to stop.\n\n' "$transcript"

# Let Voxtral flush its last words before tee exits on EOF.
trap ':' INT
uv run --python 3.12 --no-editable ${uv_options[@]+"${uv_options[@]}"} voxmlx "${options[@]}" ${args[@]+"${args[@]}"} |
    (trap '' INT; exec tee -a "$transcript")
