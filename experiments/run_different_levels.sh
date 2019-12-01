output_folder="$1"
if [[ ! -d $output_folder ]]; then
	echo "first arguement must be a folder" 2>&1
	exit
fi
echo $output_folder
shift 1

for level in $(seq -w 1 1 20); do
	python ../fuse.py --output "${output_folder}/fused_${level}.jpg" \
				   --l-max $level \
				   $@
	if [ $? -ne 0 ]; then
		break
	fi
done
