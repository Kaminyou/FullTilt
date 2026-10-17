#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to download a file and verify its SHA256 checksum
download_and_verify() {
    local file_id=$1
    local out_dir=$2
    local file_name=$3
    local expected_hash=$4
    local file_path="$out_dir/$file_name"

    # Ensure the target directory exists
    mkdir -p "$out_dir"

    # 1. Check if file already exists and matches the hash
    if [ -f "$file_path" ]; then
        echo "Checking existing file: $file_name..."
        local current_hash=$(sha256sum "$file_path" | awk '{print $1}')
        if [ "$current_hash" == "$expected_hash" ]; then
            echo "[OK] $file_name already exists and checksum matches. Skipping download."
            echo "-----------------------------------"
            return 0
        else
            echo "[WARN] $file_name exists but checksum mismatch. Re-downloading..."
        fi
    fi

    # 2. Download the file using gdown
    echo "Downloading $file_name..."
    gdown "$file_id" -O "$file_path"

    # 3. Verify the checksum after downloading
    echo "Verifying $file_name..."
    local actual_hash=$(sha256sum "$file_path" | awk '{print $1}')

    if [ "$actual_hash" == "$expected_hash" ]; then
        echo "[OK] $file_name verified successfully!"
    else
        echo "[ERROR] $file_name checksum mismatch!"
        echo "Expected: $expected_hash"
        echo "Got:      $actual_hash"
        exit 1
    fi
    echo "-----------------------------------"
}

echo "Starting setup..."
echo "-----------------------------------"

# Docker Image
download_and_verify "1mHBZTByo9eFlXv1IpzSdNNywLeF5X62J" "." "fulltilt_training.tar" "684be47579218d649f5ffccd7f0ab5d27e192c50744d691abb1f35a3d9c82424"

echo "All files downloaded and verified successfully!"