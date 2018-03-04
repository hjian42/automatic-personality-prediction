import glob
files = glob.glob("./*.res")
for file in files:
	with open(file) as f:
		print(f.read())
