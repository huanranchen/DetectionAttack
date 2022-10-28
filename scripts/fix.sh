

file=$1

awk '{ sub("\r$", ""); print }' $file > fixed_test.sh
mv fixed_test.sh $file