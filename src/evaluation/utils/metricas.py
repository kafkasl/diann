def precision(tp,fp):
	try:
		return tp/float(tp+fp)
	except:
		return 0

def recall(tp,fn):
	try:
		return tp/float(tp+fn)
	except:
		return 0

def f1score(tp,fp,fn):
	try:
		return (2*tp)/float(2*tp+fp+fn)
	except:
		return 0
