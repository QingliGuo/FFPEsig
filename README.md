# FFPEsig

FFPEsig uses FFPE signature as a noise profile to correct the observed mutation counts from a given FFPE WGS sample.

1. To run FFPEsig:
+ Download [FFPEsig.py](https://github.com/QingliGuo/FFPEsig/blob/main/FFPEsig.py)
+ Install [python 3](https://www.python.org/downloads/) (3.7.6) and import the packages required in [FFPEsig.py](https://github.com/QingliGuo/FFPEsig/blob/main/FFPEsig.py), including pandas (1.0.1), numpy (1.18.1), matplotlib (3.1.3), seaborn (0.10.1).

+ Run the command line:
```
python FFPEsig.py [--input|-i] <Path-to-the-DataFrame> [--sample|-s] <Sample_id> [--label|-l] <Unrepaired|Repaired> [--output_dir|-o] <Path-of-output-folder>
```
2. Example

```
python FFPEsig.py --input ./Data/simulated_PCAWG_FFPE_unrepaired.csv -sample ColoRect-AdenoCA::SP21528 --label Unrepaired --output_dir FFPEsig_OUTPUT
```
Or 

```
python FFPEsig.py -i ./Data/simulated_PCAWG_FFPE_unrepaired.csv -s ColoRect-AdenoCA::SP21528 -l Unrepaired -o FFPEsig_OUTPUT
```

**Note**
+ Input file, [--input|-i], must be a standard CSV format dataframe which column names are the sample IDs;
+ Sample ID, [--sample|-s], must be one of the sample IDs in Input file [--input|-i];
+ Label option, [--label|-l], must be either of them <Unrepaired|Repaired>.
+ The total running time for one sample is around 1-3 mins on a local Mac computer (3,1 GHz Intel Core i5).

# Analysis code
Here we include analysis codes and data used in our manuscript entitled "The mutational signatures of formalin fixation on the human genome".
+ [FFPE signatures discovery](https://qingliguo.github.io/FFPEsig/FFPEsig_discovery.html)
+ [Correction on simulated FFPE data](https://qingliguo.github.io/FFPEsig/Correctting_FFPEnoise_in_SimulatedFFPEs_from_PCAWG.html)
+ [WGS CRC FFPEs](https://qingliguo.github.io/FFPEsig/Correcting_FFPEnoise_in_WGS_FFPE_CRCs.html )
+ [Comparing refitted attributions of 96c and 80c sig](https://qingliguo.github.io/FFPEsig/Comparing_refitting_results_of_96c_80c_sig.html)

# Citation

Our preprint is avaiable in bioRxiv. [Check it out](https://www.biorxiv.org/content/10.1101/2021.03.11.434918v1). 
