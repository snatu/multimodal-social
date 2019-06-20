import mmsdk
from mmsdk import mmdatasdk
import time
import numpy

def myavg(intervals,features):
	final=numpy.average(features,axis=0)
	if len(final.shape)==2:
		final=numpy.average(final,axis=0)
        return final

def align():

	socialiq_no_align=mmdatasdk.mmdataset(mmdatasdk.socialiq.highlevel,"socialiq")
	#don't need these guys
	del socialiq_no_align["QA_BERT_lastlayer_binarychoice"]
	del socialiq_no_align["SOCIAL-IQ_QA_BERT_MULTIPLE_CHOICE"]
	del socialiq_no_align["SOCIAL_IQ_VGG_1FPS"]
	socialiq_no_align.align('Transcript_Raw_Chunks_BERT',collapse_functions=[myavg])
	#simple name change - now the dataset is aligned
	socialiq_aligned=socialiq_no_align
	
	socialiq_aligned.impute("Transcript_Raw_Chunks_BERT")
	socialiq_aligned.revert()
	
	deploy_files={x:x for x in socialiq_aligned.keys()}
	socialiq_aligned.deploy("./deployed",deploy_files)

if __name__=="__main__":
	align()
