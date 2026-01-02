cd ~/rcc

for folder in ccRCC pRCC chRCC; do
    echo "🧬 Converting TIFFs to PNGs in $folder ..."

    find "$folder" -type f \( -iname "*.tif" -o -iname "*.tiff" \) | while read -r file; do
        out="${file%.*}.png"
        convert "$file" "$out" && echo "✅ Converted: $file → $out"
    done

    echo "🎯 Finished converting $folder"
done
