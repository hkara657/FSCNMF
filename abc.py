import sys
print "This is the name of the script: ", sys.argv[0]
print "Number of arguments: ", len(sys.argv)
l = len(sys.argv)
for i in range(1,l):
	print '###',int(sys.argv[i])+100
