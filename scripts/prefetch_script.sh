#!/bin/bash
filename='sra_names.txt'
while read p; do 
	prefetch $p 
	fastq-dump --split-files ~/SRR/sra/$p'.sra' -O ~/SRR/files
	rm ~/SRR/sra/$p'.sra'
	java -jar ~/fastq_files/Trimmomatic-0.39/trimmomatic-0.39.jar SE ~/SRR/files/$p'_1.fastq' ~/fastq_files/trimmed_paired/'right_trimmed_'$p'_1.fq.gz' ILLUMINACLIP:adapters.fasta:2:30:10 SLIDINGWINDOW:20:20 LEADING:25 TRAILING:25 MINLEN:40
        rm ~/SRR/files/$p'_1.fastq'
	java -jar ~/fastq_files/Trimmomatic-0.39/trimmomatic-0.39.jar SE ~/SRR/files/$p'_2.fastq' ~/fastq_files/trimmed_paired/'right_trimmed_'$p'_2.fq.gz' ILLUMINACLIP:adapters.fasta:2:30:10 SLIDINGWINDOW:20:20 LEADING:25 TRAILING:25 MINLEN:40	
	rm ~/SRR/files/$p'_2.fastq'
done < $filename
