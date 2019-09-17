import glob
files = ['cAGR.res', 'cCON.res', 'cEXT.res', 'cOPN.res', 'cNEU.res']
for file in files:
	with open(file) as f:
		print(f.read())
