import sys
import csv

def main():
    input_corpus_csv_file_styled=open(sys.argv[1], newline='') #styled/neutralized test set
    csvreader_styled = csv.DictReader(input_corpus_csv_file_styled)

    column_header=sys.argv[2]

    output_file=open(sys.argv[3], 'w', newline='')
    csv_writer=csv.writer(output_file)
    header=["ID","From","train","content"]
    csv_writer.writerow(header)

    dataset_index={"tune_in":2,"train":1,"dev_in":0,"test_in_neut":3,"tune_out":4,"dev_out":5,"test_out_neut":6}
    index=0
    for row in csvreader_styled:
        author=row['author_id']
        dataset=row['set']
        snippet=row[column_header]
        output=[str(index),author,str(dataset_index[dataset]),snippet]
        csv_writer.writerow(output)
        index+=1

if __name__ == "__main__":
    main()