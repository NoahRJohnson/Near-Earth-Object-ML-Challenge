import urllib2
import requests


notRemoved = open('NotRemoved.txt', 'r')
removed = open('Removed.txt', 'r')

def assignReferers(names_file):
    name_dict = {}
    for name in names_file:
        name = name.rstrip('\n')
        name_parts = name.split()
        name_id = ""
        if len(name_parts) == 1:
            name_id = name
        else:
            name_id = name_parts[0] + "+" + name_parts[1]

        name_dict[name] = "http://www.minorplanetcenter.net/db_search/show_object?utf8=%%E2%%9C%%93&object_id=%s" % (name_id)

    return name_dict

notRemovedDict = assignReferers(notRemoved)
removedDict = assignReferers(removed)

def batchFileRequest(neo_dict, direc):
    for neo in neo_dict:
        #print neo
        name = neo.split()
        name_id = ""
        if len(name) == 1:
            name_id = name + ".txt"
        else:
            name_id = name[0] + "_" + name[1] + ".txt"

        url = 'http://www.minorplanetcenter.net/tmp/%s' % (name_id) #http://www.minorplanetcenter.net
        #print url
        s = requests.Session()
        s.get(neo_dict[neo])
        #s.headers.update({'referer':neo_dict[neo]})
        r = s.get(url)

        path = "." + direc + "%s" % (neo) + ".txt"
        with open(path, "wb+") as f:
            f.write(r.text)


        # req = urllib2.Request(url)
        # req.add_header('Referer', neo_dict[neo])
        # r = urllib2.urlopen(req)

batchFileRequest(removedDict, "/removed/")

# Referer: http://www.minorplanetcenter.net/db_search/show_object?utf8=%E2%9C%93&object_id=2016+FV7
#
# req = urllib2.Request('http://www.example.com/')
# req.add_header('Referer', 'http://www.python.org/')
# r = urllib2.urlopen(req)
