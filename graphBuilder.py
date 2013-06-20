from math import *
from copy import *

input = "yeast2HybridData.txt" #Large input file of all interactions.
MIPS = "othersPlus2.csv" #The full set of complexes
output0 = "ItoCoreNoEdge.csv" 
output1a = "ItoCoreNotConNorm.csv" 
output1b = "ItoCoreNotConHair.csv"
output1c = "ItoCoreNotConMax.csv"
outputCa = "ItoCoreConNorm.csv"
outputCb = "ItoCoreConHair.csv"
outputCc = "ItoCoreConMax.csv"

def individualCC(vList):
  min, max = 0, 2
	sum, valid = 0, 0
	for v in vList:
		#Check to see if it has a valid CC
		n = len(vList[v]) - 1
		if n > 1:
			valid = valid + 1
			#Calculate it's CC
			edges = 0
			for nbr in vList[v][1:]:
				for other in vList[v][1:]:
					if other in vList[nbr]:
						edges = edges + 1
			CC = 1.0*edges/(n*(n-1))
			sum = sum + CC
			if CC > max:
				max = CC
			if CC < min:
				min = CC

	return max, min

def printComplex(vList):
	for vertex in vList.keys():
		print vertex," ",
                print vList[vertex]
	print

#Decides if the small graph is a subgraph of the big graph
#Assumes that every vertex in small is also in big
def sub(small, big):
	for v in small.keys():
		for nbr in small[v]:
			if nbr not in big[v]:
				return False
	return True

#Given a list of values, calculate their standard deviation
def STDev(values):
	ave = 0
	for x in values:
		ave = ave + x
	ave = 1.0*ave/len(values)
	dev = 0
	for x in values:
		dev = dev + (x-ave)*(x-ave)
	dev = dev/len(values)
	return sqrt(dev)	

def clearForNext(vList):
	for vertex in vList.keys():
		if(vList[vertex][0] == 'G'):
			vList[vertex][0] = 'C'

def clearComplete(vList):
	for vertex in vList.keys():
		vList[vertex][0] = 'C'

#Generate all possible k-element subsets of a set
def kTuples(set, k):
        if len(set) < k:
                print "I think there is something wrong!"
                return []
        if len(set) == k:
                return [set]
        if k == 1:
                return [[x] for x in set]
        tupleList = []
        for i in range(len(set) - (k-1)):
                temp = kTuples(set[(i+1):], k-1)
                temp = [x + [set[i]] for x in temp]
                tupleList = tupleList + temp
        return tupleList

#Remove a set of vertices from the graph by "blacking them out"
def remove(vList, set):
	for v in set:
		vList[v][0] = 'B'

#Return all vertices to default "clear" position
def restore(vList):
	for v in vList.keys():
		vList[v][0] = 'C'


#Depth-First Search from a given vertex
def DFS(vList, start):
	vList[start][0] = 'G'
	for nbr in vList[start][1:]:
		if vList[nbr][0] == 'C':
			DFS(vList, nbr)

#Checks to see how many pieces an unconnected graph
def connectedPieces(vList):
	verts = vList.keys()
	pieces = 0
	done = False
	while not done:
		done = True
		start = 'none'
		for v in verts:			#Find a vertex not 
			if vList[v][0] == 'C':	#blacked out to use
				start = v	#as a starting point
		if start == 'none':	#If all vertices are blacked out
			restore(vList)  #then we
			return pieces	#say that graph is not connected
		pieces = pieces + 1
		DFS(vList, start)
		for v in verts:
			if vList[v][0] == 'C': #Vertex was not blacked out, nor visited
				done = False
	restore(vList)
	return pieces	

#Checks to see if a graph is connected using DFS
def isConnected(vList):
	verts = vList.keys()
	start = 'none'
	for v in verts:			#Find a vertex not 
		if vList[v][0] == 'C':	#blacked out to use
			start = v	#as a starting point
	if start == 'none':	#If all vertices are blacked out
		restore(vList)  #then we
		return False	#say that graph is not connected
	DFS(vList, start)
	for v in verts:
		if vList[v][0] == 'C': #Vertex was not blacked out, nor visited
			restore(vList)
			return False
	restore(vList)
	return True

#Returns capacities of all edges to 1
def resetCap(cap):
        for v in cap.keys():
                for edge in cap[v][1:]:
                        edge[1] = 1

#Finds a path from a source to a sink
def findFlow(s, t, cap):
        cap[s][0] = 'G'
        for nbr in cap[s][1:]:
                if nbr[1] > 0 and cap[nbr[0]][0] == 'C':
                        if nbr[0] == t:
                                return[s,t]
                        path = findFlow(nbr[0],t,cap)
                        if len(path) > 0:
                                return [s] + path
        return []

#convert the standard vList into one that can be used to find node-connect
def vCapacity(vList):
    capacity = {}
    for v in vList.keys():
	vOUT = v + "OUT"
        capacity[v] = ['C', [vOUT, 1]]
	capactiy[vOUT] = ['C', [v, 0]]
        for nbr in vList[v][1:]:
            capacity[vOUT].append([nbr, 1])
	    capacity[v].append([nbr + "OUT", 0])
    return capacity


#convert the standard vList into a capacity-holding vList
def toCapacity(vList):
    capacity = {}
    for v in vList.keys():
        capacity[v] = ['C']
        for nbr in vList[v][1:]:
            capacity[v].append([nbr, 1])
    return capacity

def vertexConnect(capacity, pairs):
    min = len(capacity)/2
    minCut = [] #Note: not sure how to get this yet...
    for pair in pairs:
        resetCap(capacity) #Reset all edge capacities to 1
	capacity[pair[0]][1][1] = len(capacity[pair[0]])#Except current
        #Do Ford-Fulkerson to find min cut
        done = False
        thisMin = 0
        while not done:
            done = True
            path = findFlow(pair[0], pair[1], capacity)
            restore(capacity)
            if len(path) > 0:
                done = False
                thisMin = thisMin + 1
                for e in range(len(path)-1):
                    #Decrease the capacity of that edge by 1
                    for nbr in capacity[path[e]][1:]:
                        if nbr[0] == path[e+1]:
                            nbr[1] = 0
                    #Increase the capacity of the reverse edge by 1
                    for nbr in capacity[path[e+1]][1:]:
                        if nbr[0] == path[e]:
                            nbr[1] = 2
	if thisMin == 0:
	    print thisMin, " goes with pair ", pair

        if thisMin < min:
            min = thisMin
            #minCut = Corresponding cut

    return min#, minCut



def edgeConnect(capacity, pairs):
    min = len(capacity)
    minCut = [] #Note: not sure how to get this yet...
    for pair in pairs:
        #print "Looking at pair ", pair
        resetCap(capacity) #Reset all edge capacities to 1
        #Do Ford-Fulkerson to find min cut
        done = False
        thisMin = 0
        while not done:
            done = True
            path = findFlow(pair[0], pair[1], capacity)
            #print "Found path", path
            #print capacity[pair[0]]
            #print capacity[pair[1]]
            restore(capacity)
            if len(path) > 0:
                #print "Length of path was positive"
                done = False
                thisMin = thisMin + 1
                for e in range(len(path)-1):
                    #Decrease the capacity of that edge by 1
                    for nbr in capacity[path[e]][1:]:
                        if nbr[0] == path[e+1]:
                            nbr[1] = 0
                    #Increase the capacity of the reverse edge by 1
                    for nbr in capacity[path[e+1]][1:]:
                        if nbr[0] == path[e]:
                            nbr[1] = 2
            #print capacity[pair[0]]
            #print capacity[pair[1]]

        if thisMin < min:
            #print thisMin, " goes with pair ", pair
            #print "From graph: "
            #print capacity
            min = thisMin
            #minCut = Corresponding cut

    #restore(vList)
    return min#, minCut


def connectivity(vList):
	if not isConnected(vList):
		return 0
	vcon = 1
	while vcon < len(vList.keys()):
		check = kTuples(vList.keys(), vcon)
		for set in check:
			remove(vList, set)
			if not isConnected(vList):
				return vcon
		vcon = vcon + 1
	return vcon-1

#Calculates the number of "stars"
#A star is a vertex where all of it's neighbors except 1 have degree 1
def countStars(vList):
	stars = []
	for v in vList.keys():
		rays = 0
		for nbr in vList[v][1:]:
			if len(vList[nbr]) == 2: #1 neighbor + indicator variable
				rays = rays + 1
		if rays == len(vList[v])-2 and rays > 0: #All but one neighbor is deg 1
			stars = stars + [v]
	return stars

#Removes all vertices of degree k or less from the graph
#then iterates until min degree of graph >= k
#If k=1, this is a "haircut," eliminating degree 1 vertices
def haircut(vList, k):
	done = False
	while not done:
		done = True
		for v in vList.keys():
			if len(vList[v]) <= k+1:
				for other in vList.keys():
					if v in vList[other]:
						vList[other].remove(v)
				del vList[v]	
				done = False

#Calculates the clustering coefficient for the graph
def clusteringCoefficient(vList):
	if len(vList.keys()) < 3:
		return "N/A"
	triples = kTuples(vList.keys(), 3)
	T = 0
	P2 = 0
	for set in triples:
		if set[0] in vList [set[1]] and set[0] in vList [set[2]]:
			 if set[1] in vList [set[2]]:
				T = T + 1
		if set[0] in vList [set[1]] and set[0] in vList [set[2]]:
			P2 = P2 + 1
		if set[1] in vList [set[0]] and set[1] in vList [set[2]]:
			P2 = P2 + 1
		if set[2] in vList [set[1]] and set[2] in vList [set[0]]:
			P2 = P2 + 1
	if P2 == 0:
		return "N\A"
	return (3.0)*T/P2

def printDist(distribution):
	for i in range(len(distribution)):
		print i, " ",
	print
	for i in distribution:
		print i, " ",
	print

#Given a distribution, turn it into a list
def distToList(dist):
	items = []
	for i in range(len(dist)):
		temp = [i] * dist[i]
		items = items + temp
	return items

def edgeCalc(vList):
	edges = 0
	for vertex in vList.keys():
		edges = edges + len(vList[vertex]) - 1
	return edges/2

def degreeCalcs(vList, n):
	max, min = 0, n #Initial values for max and min
	dist = [0] * n #Start with a distribution of all 0s
	sum = 0
	#print vList
	#print n
	for vertex in vList.keys():
		deg = len(vList[vertex]) - 1
		#print vertex, deg
		#print vList[vertex]
		if deg > max:
			max = deg
		if deg < min:
			min = deg
		sum = sum + deg
		dist[deg] = dist[deg] + 1
	return max, min, (sum*1.0)/n, dist

#Returns the intersection of the sets of neighbors of v1 and v2
def intersection(vList, v1, v2):
	set = []
	for i in vList[v1][1:]: #Don't count the indicator!
		if i in vList[v2][1:]: #Don't count the indicator!
			set.append(i)
	return set

#Calculates the average and ST Dev of the MCCs of all pairs
#Uses the Meet/Min standard for MCC
def MCCs(vList):
	pairs = kTuples(vList.keys(), 2)
	MCC = []
	ave = 0
	meaningful = 0
	for pair in pairs:
		shared = len(intersection(vList, pair[0], pair[1]))
		lp0 = len(vList[pair[0]]) - 1 #Get rid of indicator
		if pair[1] in vList[pair[0]]:
			lp0 = lp0 - 1
		lp1 = len(vList[pair[1]]) - 1 #Get rid of indicator
		if pair[1] in vList[pair[0]]:
			lp1 = lp1 - 1
		if lp0 > 0 and lp1 > 0:
			meaningful = meaningful + 1
			this = 1.0*shared/min(lp0, lp1)
			MCC.append(this)
			ave = ave + this
	if meaningful > 0:
		return ave/meaningful, STDev(MCC)
	else:
		return "N/A", "N/A"

#Given a graph, find most highly connected subgraph
def subgraph(vList):
	k = 1
	while connectivity(vList) > k:
		oldvList = deepcopy(vList)#oldvList is k+1 connected
		k = k + 1
		haircut(vList, k)
	#Check to see if there are any vertices of deg > k
	if len(vList) < 1:
		return k, oldvList
	#Get dividing k-tuple
	check = kTuples(vList.keys(), k)
	for set in check:
		remove(vList, set)
		if not isConnected(vList):
			#print "Set ", set, " divides."
			divide = set
	#print "Dividing on set ", divide
	#Use it to divide the graph
	foundV = []
	graphs = []
	for v in vList.keys():
		if v not in foundV:
			restore(vList)
			remove(vList, divide)
			DFS(vList, v)
			newGraph = {}
			for u in vList.keys():
				if vList[u][0] != 'C':
					newGraph[u] = ['C']
					for w in vList[u][1:]:
						if vList[w][0] != 'C':
							newGraph[u].append(w)
					foundV.append(u)
			graphs.append(newGraph)	
	#Check the parts
	cons = [k]
	subGs = [vList]
	for G in graphs:
		if len(G) > k+1:
			tempCon, tempG = subgraph(G)
			cons = cons + [tempCon]
			subGs = subGs + [tempG]
	max = 0
	for i in range(len(cons)):
		if cons[i] > cons[max]:
			max = i
		elif cons[i] == cons[max]:
			if len(subGs[i]) > len(subGs[max]):
				max = i
	return cons[max], subGs[max]

def biggestPiece(vList):
	foundV = []
	graphs = []
	for v in vList.keys():
		if v not in foundV:
			restore(vList)
			DFS(vList, v)
			newGraph = {}
			for u in vList.keys():
				if vList[u][0] != 'C':
					newGraph[u] = ['C']
					for w in vList[u][1:]:
						if vList[w][0] != 'C':
							newGraph[u].append(w)
					foundV.append(u)
			graphs.append(newGraph)	
	max = {}
	for G in graphs:
		if len(G) > len(max):
			max = G
	return max

def path(source, sink, vList):
	queue = []
	for v in vList[source][1:]:
		if v in sink:
			return []
		queue.append([v, [], 1])
	toReturn = []
	pathLength = len(vList)
	while len(queue) > 0:
		next = queue.pop(0)
		vList[next[0]][0] = 'G'
		for v in vList[next[0]][1:]:
			if v in sink:
				if next[2] <= pathLength:
					pathLength = next[2]
					toReturn.append(next[1]+[next[0]]) 
			elif next[2] > pathLength:
				clearComplete(vList)
				return toReturn
			elif vList[v][0] == 'C':
				queue.append([v, next[1] + [next[0]], next[2]+1])
	clearComplete(vList)
	return toReturn

def toCenter(original, vList, outfile):
	ave = 0.0
	paths = []
	for v in original.keys():
		if v not in vList.keys():
			bob =(len(path(v, vList, original)) + 1)
			ave = ave + bob
			paths.append(bob)
	outfile.write(str(max(paths))+"\t")
	outfile.write(str(ave/(len(original) - len(vList))) + "\t") 
	outfile.write(str(STDev(paths))+"\t")

def centralityData(vList,test):
	max = ave = 0
	min = len(vList)*(len(vList) - 1)
	pairs = kTuples(vList.keys(), 2)
	centrals = []
	shortest = []
	for pair in pairs:
		blah = path(pair[0], [pair[1]], vList)
		shortest.append(blah)
#		shortest.append(path(pair[0], [pair[1]], vList))
	for v in test.keys():
		central = 0
		for set in shortest:
			c = 0.0
			for p in set:
				if v in p:
					c = c + 1
			if len(set) > 0:
				c = c/len(set)
			central = central + c
		if central < min:
			min = central
		if central > max:
			max = central
		ave = ave + central
		centrals.append(central)
#		print "Vertex ", v, " has centrality ", central
	return max, min, (1.0)*ave/len(vList), STDev(centrals)

#Prints data about a graph to a string to write into the spreadsheet
def dataToString(vList):
	n = len(vList)
	m = edgeCalc(vList)
	dens = 2.0*m/(n*(n-1))
	mx, mn, ave, dist = degreeCalcs(vList, n)
	dev = STDev(distToList(dist))
	Bmx = Bmn = Bave = Bdev = "Not Calculated"
	Bmx, Bmn, Bave, Bdev = centralityData(vList,vList)
	CC = "Not Calculated"
	CC=clusteringCoefficient(vList)
	MCC = MCCdev = "Not Calculated"
	MCC, MCCdev = MCCs(vList)
	vCon = connectivity(vList)
	eCon = edgeConnect(toCapacity(vList), kTuples(vList.keys(), 2))
#	stars = countStars(vList)
	stats = str(n) + "\t" + str(m) + "\t" + str(dens) +"\t"
	stats = stats + str(mx) + "\t" + str(mn) + "\t" + str(ave) + "\t"
	stats = stats + str(dev) + "\t"+ str(CC) + "\t" + str(MCC) +"\t"
	stats = stats + str(MCCdev) + "\t"
	stats = stats + str(Bmx) + "\t" + str(Bmn) + "\t" + str(Bave) + "\t"
	stats = stats + str(Bdev)+"\t"+ str(eCon)+"\t"+str(vCon) + "\t"
	stats = stats + str(dist[1]) + "\t"
	return stats

def writeTopLine(outfile):
	outfile.write("Complex\tProteins\tType\tn\tm\t")
	outfile.write("Edge Density\tMax Degree\tMin Degree\t")
	outfile.write("Mean Degree\tDegree Deviation\tClustering Coefficient\t")
	outfile.write("Average MCC\tMCC Deviation\t")
	outfile.write("Max Between\tMin Between\tAve Between\tBetween Deviation\t")
	outfile.write("Edge Connectivity\tVertex Connectivity\tDeg 1 vertices\t")
	outfile.write("Max Path Length\tAve Path Length\tPath Length Deviation\t")
	outfile.write("Max Core Between\tMin Core Between\t")
	outfile.write("Ave Core Between\tCore Between Dev\n")



def process(ID, c):
#This must be included	outfile.write(ID+"\t"+str(len(complex))+"\t")
	error = False
	vList = {}
	for p in c:
		if p in bigGraph.keys():
			vList[p] = ['C']
			for other in c:
				if other in bigGraph[p]:
					if other in vList[p]:
						error = True
					vList[p].append(other)
	if edgeCalc(vList) == 0 or error == True:
		out0.write(ID+'\t'+str(len(c))+'\t'+"Normal\t")
		out0.write(str(len(vList))+"\t0\n")
		print "Error on ID:", ID
	else:
		if not isConnected(vList):
			#print ID, ",", connectedPieces(vList)
			norm = notNorm
			hair = notHair
			maxCon = notMax
		else:
			norm = conNorm
			hair = conHair
			maxCon = conMax
		norm.write(ID+'\t'+str(len(c))+'\t'+"Normal\t")
		norm.write(dataToString(vList)+'\n')
		original = deepcopy(vList)
		haircut(vList,1)
		if len(vList) > 1: 
			hair.write(ID+'\t'+str(len(c))+'\t'+"Haircut\t")
			hair.write(dataToString(vList))
			if len(vList) < len(original):
				toCenter(original, vList, hair)
				mx,mn,a,d =centralityData(original, vList)
				hair.write(str(mx)+"\t"+str(mn)+"\t"+str(a)+"\t"+str(d))
			hair.write("\n")
			trouble = []
			#Note: the following if statement calculates the MHCS and the value of its associated statistics. 
			#This is a time-intensive process and may be a problem for some data sets, depending on how large the complexes are.
			#If you want to calculate the MHCS and its statistics, the below statement should read, "if True:"
			#If you do not want to calculate the MHCS, the below statment should read, "if False:"
			#If you want to calculate the MHCS on some complexes but not others, put the ids of those you do not want to calculate in the list "trouble,"
			#and the below statement should read, "if ID not in trouble:"
			if False: #if ID not in trouble:	
				print "Calculating MHCS"
				k, MHCS = subgraph(vList)
				if len(MHCS) > 1:
					maxCon.write(ID+'\t'+str(len(c))+'\t'+"Max Connected\t")
					maxCon.write(dataToString(MHCS))
				if len(MHCS) < len(original):
					toCenter(original, MHCS, maxCon)
					mx,mn,a,d =centralityData(original, MHCS)
				#maxCon.write(str(mx)+"\t"+str(mn)+"\t"+str(a)+"\t"+str(d))
				maxCon.write("\n")
		#elif connectivity(original) == 0:
		else:
			MHCS = biggestPiece(original)
			if len(MHCS) > 1:
				maxCon.write(ID+'\t'+str(len(c))+'\t'+"Biggest Piece\t")
				maxCon.write(dataToString(MHCS)+"\n")

data = open(input, 'r')
#out0 = open(output0, 'w')
#out1 = open(output1, 'w')
#outC = open(outputC, 'w')

#writeTopLine(out0)
#writeTopLine(out1)
#writeTopLine(outC)

out0 = open(output0, 'w')
notNorm = open(output1a, 'w')
notHair = open(output1b, 'w')
notMax = open(output1c, 'w')
conNorm = open(outputCa, 'w')
conHair = open(outputCb, 'w')
conMax = open(outputCc, 'w')

writeTopLine(out0)
writeTopLine(notNorm)
writeTopLine(notHair)
writeTopLine(notMax)
writeTopLine(conNorm)
writeTopLine(conHair)
writeTopLine(conMax)

#The following process is used for the combined Y2H Data
bigGraph = {}
for line in data:
	lineS = line.split("\t")
	if lineS[0] not in bigGraph:
		bigGraph[lineS[0]] = ['C']
	if lineS[1] not in bigGraph[lineS[0]] and lineS[0] != lineS[1]:
		bigGraph[lineS[0]] = bigGraph[lineS[0]] + [lineS[1]]
	if lineS[1] not in bigGraph:
		bigGraph[lineS[1]] = ['C']
	if lineS[0] not in bigGraph[lineS[1]] and lineS[0] != lineS[1]:
		bigGraph[lineS[1]] = bigGraph[lineS[1]] + [lineS[0]]
data.close()

"""
#The following procedure is used for the Y2H Union dataset
bigGraph = {}
for line in data:
	lineS = line.split()
	for i in range(0,8,2):
		if lineS[i] not in bigGraph:
			bigGraph[lineS[i]] = ['C']
		if lineS[i+1] not in bigGraph[lineS[i]] and lineS[i] != lineS[i+1]:
			bigGraph[lineS[i]] = bigGraph[lineS[i]] + [lineS[i+1]]
		if lineS[i+1] not in bigGraph:
			bigGraph[lineS[i+1]] = ['C']
		if lineS[i] not in bigGraph[lineS[i+1]] and lineS[i] != lineS[i+1]:
			bigGraph[lineS[i+1]] = bigGraph[lineS[i+1]] + [lineS[i]]
data.close()
"""
"""
#The following prodedure is used for the Y2H Modified dataset
bigGraph = {}
for line in data:
	lineS = line.split()
	bigGraph[lineS[0]] = ['C']
	i = 1
	while lineS[i] != '*':
		bigGraph[lineS[0]].append(lineS[i])
		i = i+1
data.close()
"""
data = open(MIPS, 'r')

print "The graph has ", len(bigGraph), " vertices and ", edgeCalc(bigGraph), " edges.\n"

complex = []
ID = ''
for line in data:
	lineS = line.split("\t")
	if ID != lineS[0]:
		print "Starting complex", ID, "."
		if len(complex) > 0:
			process(ID, complex)
		ID = lineS[0]
		complex = []
	complex.append(lineS[2].rstrip().upper())
