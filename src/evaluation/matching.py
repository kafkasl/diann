import sys,os,re
from utils.metricas import precision,recall,f1score
from utils.brat import generate_ann
##################################################################################################
#
#	This script requires two folders:
#	-	argv[1] = goal standard / any other file
#	-	argv[2] = system annotation / any other file
#
#	The evaluation will be carried out according to the files that are in both folders.
#
#
##################################################################################################

if len(sys.argv)<3:
    print("##################################################################################################\n\
#\n\
#	This script requires two folders:\n\
#	-	argv[1] = goal standard / any other file\n\
#	-	argv[2] = system annotation / any other file\n\
#\n\
#	The evaluation will be carried out according to the files that are in both folders.\n\
#\n\
#\n\
##################################################################################################")
    sys.exit()

if not os.path.exists(sys.argv[1]):
    print("Check Goal path: ",sys.argv[1])
    sys.exit()
if not os.path.exists(sys.argv[2]):
    print("Check System path: ",sys.argv[2])
    sys.exit()


system_files = os.listdir(sys.argv[2])
gs_files     = os.listdir(sys.argv[1])

gs_text = []
system_text = []


global_system_anotations    = {
			"Disability":{"fp":0, "tp":0,"fn":0},
			"Scope":{"fp":0, "tp":0,"fn":0},
			"Neg":{"fp":0, "tp":0,"fn":0},
            "Disability+Scope+Neg":{"fp":0, "tp":0,"fn":0}				
                        }
errors = []
print("\n\n")
for fi in gs_files:
    gs_text = open(sys.argv[1]+"/"+fi,"rb").read().decode("utf-8").strip().split("\n")
    try:
        system_text = open(sys.argv[2]+"/"+fi,"rb").read().decode("utf-8").strip().split("\n")
    except:
        print("LOG: File Not found: "+sys.argv[2]+"/"+fi)
        errors.append("File Not found: "+sys.argv[2]+"/"+fi)
        continue
    if not len(gs_text)==len(system_text):
        print("LOG: Files must have the same number of lines:")
        print(sys.argv[1]+"/"+fi)
        print(sys.argv[2]+"/"+fi)
        errors.append("Files must have the same number of lines:"+sys.argv[2]+"/"+fi)
        continue

    for l in range(len(gs_text)):
        for term in ["Disability","Scope","Neg"]:
            an_disa    = [linea for linea in generate_ann(gs_text[l]).strip().split("\n")     if not linea=="" and term in linea.split("\t")[1]]
            an_system  = [linea for linea in generate_ann(system_text[l]).strip().split("\n") if not linea=="" and term in linea.split("\t")[1]]
            for an in list(an_disa):
                if an in an_system:
                    an_system.remove(an)
                    an_disa.remove(an)
                    global_system_anotations[term]["tp"]+=1
                else:
                    global_system_anotations[term]["fn"]+=1
            global_system_anotations[term]["fp"]+=len(an_system)
        
        an_disa   = re.findall(r'(\<scp\>(.+?)\<\/scp\>)',gs_text[l])
        an_system = re.findall(r'(\<scp\>(.+?)\<\/scp\>)',system_text[l])
        for an in list(an_disa):
            if an in an_system:
               an_system.remove(an)
               an_disa.remove(an)
               global_system_anotations["Disability+Scope+Neg"]["tp"]+=1
            else:
               global_system_anotations["Disability+Scope+Neg"]["fn"]+=1
        global_system_anotations["Disability+Scope+Neg"]["fp"]+=len(an_system)


print("\n\n\nResults:")
for x in global_system_anotations.keys():
    print("=========================================================")
    print(x+":")
    print("---------------------------------------------------------")
    print(global_system_anotations[x])
    print("Precision:",precision(global_system_anotations[x]["tp"],global_system_anotations[x]["fp"]))
    print("Recall:",   recall(global_system_anotations[x]["tp"],global_system_anotations[x]["fn"]))
    print("F1 score:",f1score(global_system_anotations[x]["tp"],global_system_anotations[x]["fp"],global_system_anotations[x]["fn"]))
    print("=========================================================")

if not len(errors)==0:
    print(str(len(errors))+" files not evaluated.")
    print("\n- ".join(errors))


