# FFPEsig

FFPEsig uses FFPE signature as a noise profile to correct the observed mutation counts from a given FFPE WGS sample.

1. To run FFPEsig:
+ Download [FFPEsig.py](https://github.com/QingliGuo/FFPEsig/blob/main/FFPEsig.py)
+ Install [python 3](https://www.python.org/downloads/) and the packages required in [FFPEsig.py](https://github.com/QingliGuo/FFPEsig/blob/main/FFPEsig.py), including pandas, numpy, matplotlib, seaborn.
+ Run the command line:
```
python FFPEsig.py [--input|-i] <Path-to-the-DataFrame> [--sample|-s] <Sample_id> [--label|-l] <Unrepaired|Repaired> [--output_dir|-o] <Path-of-output-folder>
```
2. Example
```
python FFPEsig.py --input ./Data/simulated_PCAWG_FFPE_unrepaired.csv --sample ColoRect-AdenoCA::SP21528 --label Unrepaired --output_dir FFPEsig_OUTPUT```
```
Or 

```
python FFPEsig.py -i ./Data/simulated_PCAWG_FFPE_unrepaired.csv -s ColoRect-AdenoCA::SP21528 -l Unrepaired -o FFPEsig_OUTPUT```
```

**Note**
+ __<Path-to-the-DataFrame> file__ must be a standard CSV format dataframe which contains columns specifying sample IDs
+ <Sample_id> must be one of the sample IDs in <Path-to-the-DataFrame> file
+ Label option([--label|-l]) must be either of them <Unrepaired|Repaired>.

# Anlysis code
Here we include analysis codes and data used in our manuscript entitled "The mutational signatures of formalin fixation on the human genome".
+ [FFPE signatures discovery](https://qingliguo.github.io/FFPEsig/FFPEsig_discovery.html)
+ [Correction on simulated FFPE data](https://qingliguo.github.io/FFPEsig/Correctting_FFPEnoise_in_SimulatedFFPEs_from_PCAWG.html)
+ [WGS CRC FFPEs](https://qingliguo.github.io/FFPEsig/Correcting_FFPEnoise_in_WGS_FFPE_CRCs.html )
+ [Comparing refitted attributions of 96c and 80c sig](https://qingliguo.github.io/FFPEsig/Comparing_refitting_results_of_96c_80c_sig.html)

# Citation

To be decided: The mutational signatures of formalin fixation on the human genome

