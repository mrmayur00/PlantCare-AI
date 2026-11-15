#!/usr/bin/env bash
# get_datasets.sh - Download every image dataset used to train PlantCare AI.
#
# Creates ./sources/ (about 6 GB while downloading, ~4 GB afterwards) with:
#   plantvillage/   PlantVillage colour images (38 classes)
#   mangoleafbd/    MangoLeafBD (8 mango classes)
#   rice/           Rice diseases dataset (Healthy, BrownSpot, Hispa, LeafBlast)
#   wheat/          Wheat disease images (Healthy Wheat, Leaf Rust, ...)
#   tulsi/          Tulsi (holy basil) healthy / unhealthy dataset
#   other/          leaves of other plants, for the "unsupported plant" class
#
# Usage (Linux, macOS, Google Colab, or Git Bash on Windows):
#   bash get_datasets.sh
#   export PLANTCARE_SOURCES="$PWD/sources"
#   python prepare_dataset.py
#   python train_model.py --data-dir data/train --alpha 0.35 --image-size 128 --weights none ...
set -e
mkdir -p sources && cd sources

# Sparse clone: download only the folder we need from a GitHub repository
fetch() {  # fetch <owner/repo> <folder in repo> <target name>
  local repo=$1 path=$2 target=$3 tmp=".tmp_${3}"
  if [ -e "$target" ]; then echo "  $target already present"; return; fi
  echo "Downloading $target from github.com/$repo ..."
  rm -rf "$tmp"
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$repo" "$tmp"
  git -C "$tmp" sparse-checkout set "$path"
  mv "$tmp/$path" "$target"
  rm -rf "$tmp"
}

fetch spMohanty/PlantVillage-Dataset raw/color plantvillage
fetch DevyanshMalhotra/MangoLeafDiseaseDetection dataset mangoleafbd
fetch suhasmaddali/Wheat-Disease-Detection- Images wheat
fetch yassa-life/Basil_Leaf_health_check_dataset-_trained_model-evidence data/raw tulsi
fetch waqas-ali/tf_plant_disease_classification data other

if [ ! -e rice ]; then
  echo "Downloading rice from github.com/jonathanrjpereira/Rice-Disease-Classification ..."
  rm -rf .tmp_rice
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/jonathanrjpereira/Rice-Disease-Classification .tmp_rice
  git -C .tmp_rice sparse-checkout set --no-cone /Rice_All.zip
  python -c "import zipfile; zipfile.ZipFile('.tmp_rice/Rice_All.zip').extractall('.tmp_rice')"
  mv .tmp_rice/Rice_All rice
  rm -rf .tmp_rice
fi

echo
echo "Done. Now run:"
echo "  export PLANTCARE_SOURCES=\"$PWD\""
echo "  python prepare_dataset.py"
