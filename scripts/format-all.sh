FOLDERS=(sources bin)
EXTENSIONS=(h hpp hxx c m cpp cxx mm glsl hlsl cu)

for folder in ${FOLDERS[@]}
do
  for ext in ${EXTENSIONS[@]}
  do
    find "./${folder}" -type f -iname "*.${ext}" -exec ./scripts/format-file.sh {} \;
  done
done
