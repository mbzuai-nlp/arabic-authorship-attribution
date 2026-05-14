
mkdir -p downloaded_books
./download_data_hindawi_gumar.sh

muddler unmuddle \
  -s ./downloaded_books \
  -m ./dataset_A3D.muddled \
  dataset_A3D.csv

muddler unmuddle \
  -s /Users/injy.hamed/Documents/Projects/Arabic_AA_TST/camera_ready/data/muddler/downloaded_books_encrypt \
  -m ./dataset_A3D.muddled \
  dataset_A3D.csv
