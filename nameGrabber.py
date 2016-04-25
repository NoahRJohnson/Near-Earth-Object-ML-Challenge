from HTMLParser import HTMLParser
import urllib2

response1 = urllib2.urlopen('http://neo.jpl.nasa.gov/risk/')
html1 = response1.read()

response2 = urllib2.urlopen('http://neo.jpl.nasa.gov/risk/removed.html')
html2 = response2.read()

# I didn't have enough time to make it one class so yolo
class RemovedParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.grabData = False
        self.inTable = False
        self.tthits = 0
        self.laststarttag = ''
        self.lastendtag = ''
        self.riskNames = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            for attr in attrs:
                if attr[0] == 'cellpadding' and attr[1] == '5':
                    self.inTable = True

        if self.inTable == True:
            if tag == 'tt':
                self.tthits += 1
                if self.tthits % 2 == 0:
                    self.grabData=True

    def handle_endtag(self, tag):
        if tag == 'table':
            self.inTable = False

    def handle_data(self,data):
        if self.grabData == True:
            if '(' in data:
                start = data.find('(') + 1
                end = data.find(')')
                self.riskNames.append(data[start:end])

            else:
                self.riskNames.append(data)
            self.grabData = False
    def getNames(self):
        return self.riskNames

class RiskParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.grabData = False
        self.inTable = False
        self.tthits = 0
        self.laststarttag = ''
        self.lastendtag = ''
        self.riskNames = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.inTable = True
        if self.inTable == True:
            if tag == 'tt':
                self.tthits += 1
                if self.tthits % 10 == 1:
                    self.grabData=True

    def handle_endtag(self, tag):
        if tag == 'table':
            self.inTable = False

    def handle_data(self,data):
        if self.grabData == True:
            if '(' in data:
                start = data.find('(') + 1
                end = data.find(')')
                self.riskNames.append(data[start:end])

            else:
                self.riskNames.append(data)
            self.grabData = False
    def getNames(self):
        return self.riskNames

print "Risk objects names:\n"
parser1 = RiskParser()
parser1.feed(html1)
print parser1.getNames()


print "Removed objects names:\n"
parser2 = RemovedParser()
parser2.feed(html2)
print parser2.getNames()

with open('NotRemoved.txt', 'a') as the_file:
    for name in parser1.getNames():
        the_file.write(name + '\n')

with open('Removed.txt', 'a') as the_file:
    for name in parser2.getNames():
        the_file.write(name + '\n')
