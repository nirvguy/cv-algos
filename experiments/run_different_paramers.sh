output_folder="$1"
if [[ ! -d $output_folder ]]; then
	echo "first arguement must be a folder" 2>&1
	exit
fi
echo $output_folder
shift 1

for w_c in 0 1; do
	for w_s in 0 1; do
		for w_e in 0 1; do
			python ../fuse.py --output "${output_folder}/fused_${w_c}${w_s}${w_e}.jpg" \
						   --w-c ${w_c} --w-s ${w_s} --w-e ${w_e} \
						   $@
		done
	done
done
