# FFPEsig

FFPEsig uses FFPE signature as a noise profile to correct the observed mutation counts from a given FFPE WGS sample.

1. To run FFPEsig:
+ Download [FFPEsig.py](https://github.com/QingliGuo/FFPEsig/blob/main/FFPEsig.py)
+ Install [python 3](https://www.python.org/downloads/) and the packages required in [FFPEsig.py](https://github.com/QingliGuo/FFPEsig/blob/main/FFPEsig.py)
+ Run the command line:
```
python FFPEsig.py <path-to-sample-file> <sample-ID> <Repaired/Unrepaired>
```
2. Example
```
python FFPEsig.py ./Data/simulated_PCAWG_FFPE_unrepaired.csv ColoRect-AdenoCA::SP21528 Unrepaired
```
 
# Anlysis code
Here we include analysis codes and data used in our manuscript entitled "The mutational signatures of formalin fixation on the human genome".
+ [FFPE signatures discovery](https://qingliguo.github.io/FFPEsig/FFPEsig_discovery.html)
+ [Correction on simulated FFPE data](https://qingliguo.github.io/FFPEsig/Correctting_FFPEnoise_in_SimulatedFFPEs_from_PCAWG.html)
+ [WGS CRC FFPEs](https://qingliguo.github.io/FFPEsig/Correcting_FFPEnoise_in_WGS_FFPE_CRCs.html )


# Citation

To be decided: The mutational signatures of formalin fixation on the human genome

