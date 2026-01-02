cd ~/rcc/ccRCC
mkdir -p ccRCC_normals
find . -type f | grep -E 'TCGA-[A-Z0-9]+-[A-Z0-9]+-11[A-Z]-' | while read -r file; do
    cp "$file" ccRCC_normals/
done

cd ~/rcc/pRCC
mkdir -p pRCC_normals
find . -type f | grep -E 'TCGA-[A-Z0-9]+-[A-Z0-9]+-11[A-Z]-' | while read -r file; do
    cp "$file" pRCC_normals/
done

cd ~/rcc/chRCC
mkdir -p chRCC_normals
find . -type f | grep -E 'TCGA-[A-Z0-9]+-[A-Z0-9]+-11[A-Z]-' | while read -r file; do
    cp "$file" chRCC_normals/
done
