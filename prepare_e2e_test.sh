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

# Weights
download_and_verify "1fd-ptpz9u8LN4Yvg2JS3M2lCfur80eRB" "weights" "fulltilt.pth" "78510ae313dd70d305dd870d0536bd6a2396d7f4a19ce60a9622f8d4cef9bf27"

# Data
download_and_verify "1Ezp4aTB73b2VJDLuxWAtoo39eEcQWOU0" "data" "test_expected_output.pkl" "40e8148b8fb37d9a72e8237d03e89c155c2a31d7a30d956c0dcb98b9191e804c"
download_and_verify "1CPngxOBeGg2mtN17ReNb4DfGNslwGfGR" "data" "test_tiltseries.npy" "112c25e79c1472479a4960f6be75cabb1fc58b3448bad788d853e5d9bcc242c2"
download_and_verify "1o5F7jLDLuVrD-aNXIbIiT0GJhxwUgT6H" "data" "test_tomogram.npy" "fbbdaf2daaa1dc20d083b30e57394e434ff08e677fa1b806f744e8c92d4b998f"
download_and_verify "1Yb1rYhEWLzXo_6GqvtcaHNfOwvVPqwI5" "data" "test_cross_expected_output.pkl" "bc90e5c279279c8c57b9a104c67a5ae67616ce120b5b9bac19829d4026658ee9"
download_and_verify "1uOTf4vd9rSO0daY8qEBsmGCXdqnbTrX-" "data" "test_tiltseries_cross.npy" "e6e88a8fe1ffcfd141e53c485b4ea174502e50cc7a0bbde8d880391cdddb55c9"
download_and_verify "1CBVRka0Om0vtOdFwPrRCQkSdfO9pGqku" "data" "test_tomogram_cross.npy" "ab8d851a9ad8d33d920ba65b67c2577f5f9ac57c5886a8edbdddee35a66a3161"

echo "All files downloaded and verified successfully!"
