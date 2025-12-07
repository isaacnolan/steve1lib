#!/usr/bin/env bash

# Small helper script to download model weights and data for STEVE-1
# It will attempt to install 'gdown' via pip (user install) if missing.

# Create the directory structure if it doesn't exist
mkdir -p data/weights/vpt
mkdir -p data/weights/mineclip
mkdir -p data/weights/steve1
mkdir -p data/visual_prompt_embeds
mkdir -p data/prior_dataset

# Ensure gdown is available (used to download from Google Drive). Try to auto-install if missing.
if ! command -v gdown >/dev/null 2>&1; then
	echo "gdown not found. Attempting to install via pip (user install)..."
	if command -v python3 >/dev/null 2>&1; then
		python3 -m pip install --user --upgrade gdown || true
	elif command -v python >/dev/null 2>&1; then
		python -m pip install --user --upgrade gdown || true
	else
		echo "No python executable found to install gdown. Please install gdown manually."
	fi

	if ! command -v gdown >/dev/null 2>&1; then
		echo "gdown still not available. You can install it with:\n  pip install --user gdown\nor via conda:\n  conda install -c conda-forge gdown"
		echo "Proceeding — some downloads may fail."
	fi
fi

# Check for unzip
if ! command -v unzip >/dev/null 2>&1; then
	echo "unzip not found. The visual prompt embeds zip file can't be extracted automatically. Please install 'unzip' or extract 'data/visual_prompt_embeds.zip' manually."
fi

# Base models for VPT
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-foundation-2x.weights -P data/weights/vpt
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model -P data/weights/vpt

# MineCLIP weights
download_from_gdrive() {
	# Usage: download_from_gdrive FILEID OUTPUT_PATH
	FILEID="$1"
	OUTPATH="$2"

	# Try gdown first
	if command -v gdown >/dev/null 2>&1; then
		echo "Downloading $OUTPATH via gdown..."
		gdown "https://drive.google.com/uc?id=${FILEID}" -O "${OUTPATH}" && return 0
	fi

	# Fallback to wget/curl that handles Google Drive confirm tokens
	echo "gdown not available or failed; attempting wget fallback for $OUTPATH..."
	TMPCOOKIES="/tmp/gdrive_cookies_$$.txt"
	CONF_URL="https://drive.google.com/uc?export=download&id=${FILEID}"

	# First request to get the confirm token
	HTML=$(wget --quiet --save-cookies ${TMPCOOKIES} --keep-session-cookies --no-check-certificate "${CONF_URL}" -O - || true)
	CONFIRM=$(echo "${HTML}" | sed -n 's/.*confirm=\([0-9A-Za-z_-]*\).*/\1/p' | head -n1)

	if [ -z "${CONFIRM}" ]; then
		# Try alternate pattern
		CONFIRM=$(echo "${HTML}" | grep -o 'confirm=[0-9A-Za-z_-]*' | sed 's/confirm=//' | head -n1 || true)
	fi

	if [ -n "${CONFIRM}" ]; then
		wget --load-cookies ${TMPCOOKIES} "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILEID}" -O "${OUTPATH}" || true
	else
		# Last resort: try curl streaming
		curl -L -o "${OUTPATH}" "${CONF_URL}" || true
	fi

	rm -f ${TMPCOOKIES}
}

download_from_gdrive 1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW data/weights/mineclip/attn.pth

# STEVE-1 weights
download_from_gdrive 1E3fd_-H1rRZqMkUKHfiMhx-ppLLehQPI data/weights/steve1/steve1.weights

# Prior weights
download_from_gdrive 1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES data/weights/steve1/steve1_prior.pt

# Prior dataset
download_from_gdrive 18JKzIwHmFBrAjfiRNobtwkN7zhwQc7IO data/prior_dataset/data.pkl

# Download visual prompt embeds
download_from_gdrive 1K--DOHMDKjtklTK6SbpH11wrI2j_61mu data/visual_prompt_embeds.zip || true
if [ -f data/visual_prompt_embeds.zip ]; then
	if command -v unzip >/dev/null 2>&1; then
		unzip -o data/visual_prompt_embeds.zip -d data
	else
		echo "Downloaded data/visual_prompt_embeds.zip but 'unzip' is not available to extract it."
	fi
else
	echo "Failed to download data/visual_prompt_embeds.zip. Check network or download manually into data/visual_prompt_embeds.zip"
fi