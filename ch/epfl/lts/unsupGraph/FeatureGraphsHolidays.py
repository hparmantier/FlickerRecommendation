
from Graph import Graph
from HolidaysReader import HolidaysReader

#CREATE GRAPH FILES FROM IMAGES FEATURES
def main():
    file = 'D:\cours\MA1\Semester Project\datasets\holidays\outputs\graphs\\'
    reader = HolidaysReader()
    reader.extractData()
    datas = reader.all()
    for k,v in datas.iteritems():
        g = Graph.fromData(v)
        g.buildNNGraph(5)
        g.buildNXGraph()
        g.nnSave(file+k+'-nn')
        g.nxSave(file+k+'-nx')




main()