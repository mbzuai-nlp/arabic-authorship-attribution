import sys
import csv

def main():
    input_corpus_csv_file=open(sys.argv[1], newline='')
    csvreader = csv.DictReader(input_corpus_csv_file)

    output_file=open(sys.argv[2], 'w', newline='')
    csv_writer=csv.writer(output_file)
    header=["ID","From","train","content"]
    csv_writer.writerow(header)

    dataset_index={"tune_in":2,"train":1,"dev_in":0,"test_in":3,"tune_out":4,"dev_out":5,"test_out":6}
    index=0
    for row in csvreader:
        author=row['author_id']
        dataset=row['set']
        snippet=row['content']
        #rest_zeros=[str(0)]*55
        output=[str(index),author,str(dataset_index[dataset]),snippet]
        #output.extend(rest_zeros)
        csv_writer.writerow(output)
        index+=1

if __name__ == "__main__":
    main()