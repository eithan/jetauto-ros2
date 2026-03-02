#!/usr/bin/env bash
# Download test images for jetauto_sim from Pexels CDN.
# Run from the package root: bash download_images.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGES_DIR="$SCRIPT_DIR/images"
mkdir -p "$IMAGES_DIR"

echo "Downloading test images to: $IMAGES_DIR"
echo ""

dl() {
    local url="$1"
    local file="$IMAGES_DIR/$2"
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo "  ✓ $2 (exists)"
        return
    fi
    echo "  ↓ $2 ..."
    if curl -L -s --max-time 15 -o "$file" "$url" && [ -s "$file" ]; then
        echo "    ✓ done ($(du -h "$file" | cut -f1))"
    else
        echo "    ✗ FAILED — $2 will be skipped in sim"
        rm -f "$file"
    fi
}

# ── Single subjects ──────────────────────────────────────────────────────── #
dl "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?w=640&h=480&fit=crop&crop=face"  "01_man.jpg"
dl "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?w=640&h=480&fit=crop&crop=face"  "02_woman.jpg"
dl "https://images.pexels.com/photos/35537/child-children-girl-happy.jpg?w=640&h=480&fit=crop"        "03_child.jpg"
dl "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?w=640&h=480&fit=crop"          "04_dog.jpg"
dl "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?w=640&h=480&fit=crop"      "05_cat.jpg"
dl "https://images.pexels.com/photos/326900/pexels-photo-326900.jpeg?w=640&h=480&fit=crop"            "06_bird.jpg"
dl "https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg?w=640&h=480&fit=crop"            "07_car.jpg"
dl "https://images.pexels.com/photos/1080696/pexels-photo-1080696.jpeg?w=640&h=480&fit=crop"          "08_table.jpg"
dl "https://images.pexels.com/photos/1350789/pexels-photo-1350789.jpeg?w=640&h=480&fit=crop"          "09_chair.jpg"

# ── Mixed scenes ─────────────────────────────────────────────────────────── #
dl "https://images.pexels.com/photos/1024311/pexels-photo-1024311.jpeg?w=640&h=480&fit=crop"          "10_man_woman.jpg"
dl "https://images.pexels.com/photos/1128318/pexels-photo-1128318.jpeg?w=640&h=480&fit=crop"          "11_family.jpg"
dl "https://images.pexels.com/photos/1404819/pexels-photo-1404819.jpeg?w=640&h=480&fit=crop"          "12_dog_cat.jpg"
dl "https://images.pexels.com/photos/2023384/pexels-photo-2023384.jpeg?w=640&h=480&fit=crop"          "13_dog_bird.jpg"
dl "https://images.pexels.com/photos/1393619/pexels-photo-1393619.jpeg?w=640&h=480&fit=crop"          "14_person_car.jpg"
dl "https://images.pexels.com/photos/1080721/pexels-photo-1080721.jpeg?w=640&h=480&fit=crop"          "15_table_chair.jpg"
dl "https://images.pexels.com/photos/1267320/pexels-photo-1267320.jpeg?w=640&h=480&fit=crop"          "16_person_table.jpg"
dl "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?w=640&h=480&fit=crop"          "17_pets.jpg"
dl "https://images.pexels.com/photos/1753706/pexels-photo-1753706.jpeg?w=640&h=480&fit=crop"          "18_person_dog.jpg"
dl "https://images.pexels.com/photos/1703192/pexels-photo-1703192.jpeg?w=640&h=480&fit=crop"          "19_street_scene.jpg"
dl "https://images.pexels.com/photos/1683975/pexels-photo-1683975.jpeg?w=640&h=480&fit=crop"          "20_complex.jpg"

echo ""
echo "Download complete!"
echo ""
ls -lh "$IMAGES_DIR"
