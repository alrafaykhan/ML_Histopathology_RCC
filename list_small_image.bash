cd ~/rcc

for folder in ccRCC pRCC chRCC; do
    echo "🔍 Checking low-quality images in $folder ..."

    find "$folder" -type f -iregex '.*\.png$' -print0 \
    | xargs -0 -I{} identify -format '%w %h %i\n' "{}" \
    | awk '$1<1000 || $2<1000 {print $3}' \
    | while read -r img; do
        echo "🗑️ Deleting low-quality: $img"
        rm -f "$img"
    done

    echo "✅ Finished cleaning $folder"
done
