#!/usr/bin/python
#!coding=utf-8

tag_index={}
count=0
for line in open("data/testdata","r"):
	li=line.split("\t")
	if len(li)==2:
		if li[1].strip()=="":
			continue
		else:
			tags=li[1].split(",")
			for tag in tags:
				tag=tag.strip()
				if tag!="":
					if not tag_index.has_key(tag):
						tag_index[tag]=count
						count=count+1


fo=open("data/indextag.txt","w")
for tag,index in sorted(tag_index.iteritems(),key=lambda(k,v):(v,k)):
	fo.write(str(index)+"\t"+tag+"\n")
fo.close()


fout=open("data/formattags.txt","w")
for line in open("data/testdata","r"):
	li=line.split("\t")
	if len(li)==2:
		if  li[1].strip()!="":
			strlist=[]
			strlist.append(li[0])
			tags=li[1].split(",")
			tagindex_count={}
			for tag in tags:
				tag=tag.strip()
				if tag!="":
					index=tag_index[tag]
					tagindex_count[index]=tagindex_count.get(index,0)+1
			for tagindex,count in sorted(tagindex_count.iteritems()):
				strlist.append(str(tagindex)+":"+str(count))
			fout.write("\t".join(strlist)+"\n")
fout.close()
