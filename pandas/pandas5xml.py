# 웹 문서 읽기 : XML
import xml.etree.ElementTree as etree

'''
xmlf=open("../testdata_utf8/my.xml",mode="r",encoding="utf-8").read()
print(xmlf,type(xmlf)) #<class str>
root=etree.fromstring(xmlf)
print(root,type(root)) #<class 'xml.etree.ElementTree.Element'>
print(root.tag,' ',len(root))
print()
'''
xmlfile=etree.parse("../testdata_utf8/my.xml")
print(xmlfile,type(xmlfile)) #<class 'xml.etree.ElementTree.ElementTree'> ->노드가 생김-부모자식 형성
root=xmlfile.getroot() 
print(root.tag)
print(root[0][0].tag)
print(root[0][1].tag)
print(root[0][0].attrib) #{'id': 'ks1'} -> dict 형식으로 잡힌다.
print(root[0][0].attrib.keys())
print(root[0][0].attrib.values())
print()
myname=root.find('item').find('name').text
print(myname)
print()
for child in root:
    print(child.tag)
    for child2 in child:
        print(child2.tag,child2.attrib)
print()
children=root.findall('item')
print(children)
for it in children:
    re_id=it.find('name').get('id')
    re_name=it.find('name').text
    re_tel=it.find('tel').text
    print(re_id,re_name,re_tel)

    